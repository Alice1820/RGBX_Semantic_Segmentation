from ast import parse
import os.path as osp
import os
import sys
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel, DataParallel

from dataloader.dataloader import get_train_loader
from models.dual_builder import RGBXEncoderDecoder as dualsegmodel
from models.builder import EncoderDecoder as segmodel
from models.multimatch import rgbdFusMultiMatch, rgbdMultiMatch
from models.criterion import FixMatchLoss, MultiMatchLoss


from dataloader.RGBXDataset import RGBX_U, RGBX_X, RGBX_Base
from utils.init_func import init_weight, group_weight
from utils.lr_policy import WarmUpPolyLR
from utils.utils import AverageMeter
from engine.engine import Engine
from engine.logger import get_logger
from utils.pyt_utils import all_reduce_tensor, extant_file

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
logger = get_logger()

from sys import path
path.append('./config/')

os.environ['MASTER_PORT'] = '169710'

global best_miou
best_miou = 0

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

def create_model(config, criterion, norm_layer=None):
    if config.algo == 'supervised':
        if config.modals in ['RGB', 'Depth']:
            return segmodel(cfg=config, criterion=criterion, norm_layer=norm_layer)
        elif config.modals == 'RGBD':
            return dualsegmodel(cfg=config, criterion=criterion, norm_layer=norm_layer)
    elif config.algo == 'multimatch':
        assert config.modals == 'RGBD'
        return rgbdFusMultiMatch(config, criterion=MultiMatchLoss(ignore_index=config.background), norm_layer=norm_layer)
    elif config.algo == 'fixmatch':
        assert config.modals == 'RGBD'
        return rgbdFusMultiMatch(config, criterion=FixMatchLoss(ignore_index=config.background), norm_layer=norm_layer)
    else:
        raise NotImplementedError

def find_loss(model, config, imgs, modal_xs, gts):
    if config.algo == 'supervised':
        if config.modals == 'RGBD':
            loss = model(imgs, modal_xs, gts)
        elif config.modals == 'RGB':
            loss = model(imgs, gts)
        elif config.modals == 'Depth':
            loss = model(modal_xs, gts)
    elif config.algo == 'multimatch':
        # print (imgs.size())
        # print (modal_xs.size())
        # print (gts.size())
        loss = model(imgs, modal_xs, gts)
    return loss

def meter_outputs(METERS, outputs):
    for key, value in outputs.items():
        METERS[key].update(value.item())

def logger_outputs(METERS, tb, epoch):
    for key, value in METERS.items():
        tb.add_scalar('train/{}'.format(key), value.avg, epoch)

if __name__ == '__main__':
    # from config import config
    # assert len(sys.argv) > 1
    from importlib import import_module
    # config_name = sys.argv[1]
    # config = import_module(config_name)

    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('-d', '--devices', default='0',
                       help='set data parallel training')
    parser.add_argument('-c', '--continue', type=extant_file,
                    metavar="FILE",
                    dest="continue_fpath",
                    help='continue from one certain checkpoint')
    parser.add_argument('--local_rank', default=0, type=int,
                    help='process rank on node')
    parser.add_argument('-p', '--port', type=str,
                    default='16005',
                    dest="port",
                    help='port for init_process_group')
    '''Evaluation'''
    parser.add_argument('-e', '--epochs', default='last', type=str)
    # p.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', default='/data0/xfzhang/data/NYUv2/results/')
    args = parser.parse_args()
    config = import_module(args.config)
    # print (config)
    # from config import config
    config = config.config
    # print (config)
    # exit()

    with Engine(args) as engine:

        cudnn.benchmark = True
        seed = config.seed
        if engine.distributed:
            seed = engine.local_rank
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

         # data loader
        train_labeled_loader, train_labeled_sampler = get_train_loader(engine, RGBX_X, config, config.batch_size)
        train_unlabeled_loader, train_unlabeled_sampler = get_train_loader(engine, RGBX_Base, config, config.batch_size*config.mu)
        
        if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
            tb_dir = config.tb_dir + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
            generate_tb_dir = config.tb_dir + '/tb'
            tb = SummaryWriter(log_dir=tb_dir)
            engine.link_tb(tb_dir, generate_tb_dir)

        # config network and criterion
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=config.background)

        if engine.distributed:
            BatchNorm2d = nn.SyncBatchNorm
        else:
            BatchNorm2d = nn.BatchNorm2d
        
        model = create_model(config=config, criterion=criterion, norm_layer=BatchNorm2d)

        # group weight and config optimizer
        base_lr = config.lr
        if engine.distributed:
            base_lr = config.lr
        
        params_list = []
        params_list = group_weight(params_list, model, BatchNorm2d, base_lr)
        
        if config.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(params_list, lr=base_lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)
        elif config.optimizer == 'SGDM':
            optimizer = torch.optim.SGD(params_list, lr=base_lr, momentum=config.momentum, weight_decay=config.weight_decay)
        else:
            raise NotImplementedError

        # config lr policy
        total_iteration = config.nepochs * config.niters_per_epoch
        lr_policy = WarmUpPolyLR(base_lr, config.lr_power, total_iteration, config.niters_per_epoch * config.warm_up_epoch)

        if engine.distributed:
            logger.info('.............distributed training.............')
            if torch.cuda.is_available():
                model.cuda()
                model = DistributedDataParallel(model, device_ids=[engine.local_rank], 
                                                output_device=engine.local_rank, find_unused_parameters=False)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            # model = DataParallel(model)

        engine.register_state(dataloader=train_labeled_loader, model=model,
                            optimizer=optimizer)
        if engine.continue_state_object:
            engine.restore_checkpoint()

        optimizer.zero_grad()
        model.train()
        logger.info('begin trainning:')
        
        for epoch in range(engine.state.epoch, config.nepochs+1):
            METERS = {'loss': AverageMeter(), 'loss_x': AverageMeter(), 'loss_u': AverageMeter(), 
            'mask_rgb': AverageMeter(), 'mask_dep': AverageMeter(), 'mask_en': AverageMeter(),
            'thres_rgb': AverageMeter(), 'thres_dep': AverageMeter(), 'thres_en': AverageMeter()}

            if engine.distributed:
                train_labeled_sampler.set_epoch(epoch)
                train_unlabeled_sampler.set_epoch(epoch)
            bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
            pbar = tqdm(range(config.niters_per_epoch), file=sys.stdout,
                        bar_format=bar_format)
            labeled_iter = iter(train_labeled_loader)
            unlabeled_iter = iter(train_unlabeled_loader)
            sum_loss = 0

            for idx in pbar:
                engine.update_iteration(epoch, idx)

                minibatch_x = labeled_iter.next()
                minibatch_u = unlabeled_iter.next()

                imgs_x = minibatch_x['data'].cuda(non_blocking=True)
                gts_x = minibatch_x['label'].cuda(non_blocking=True)
                modal_xs_x = minibatch_x['modal_x'].cuda(non_blocking=True)

                imgs_u = minibatch_u['data'].cuda(non_blocking=True)
                modal_xs_u = minibatch_u['modal_x'].cuda(non_blocking=True)

                imgs = interleave(
                    torch.cat((imgs_x, imgs_u)), config.mu+1)
                modal_xs = interleave(
                    torch.cat((modal_xs_x, modal_xs_u)), config.mu+1)
                aux_rate = 0.2
                outputs = find_loss(model, config, imgs, modal_xs, gts_x)
                meter_outputs(METERS, outputs)
                loss = outputs['loss']
                # parallel training
                loss = loss.mean()
                # reduce the whole loss over multi-gpu
                if engine.distributed:
                    reduce_loss = all_reduce_tensor(loss, world_size=engine.world_size)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                current_idx = (epoch- 1) * config.niters_per_epoch + idx 
                lr = lr_policy.get_lr(current_idx)

                for i in range(len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr

                # if engine.distributed:
                #     sum_loss += reduce_loss.item()
                #     print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                #             + ' Modals: {}'.format(config.modals) \
                #             + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                #             + ' lr=%.4e' % lr \
                #             + ' loss=%.4f total_loss=%.4f' % (reduce_loss.item(), (sum_loss / (idx + 1)))
                # else:
                #     sum_loss += loss
                #     print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                #             + ' Modals: {}'.format(config.modals) \
                #             + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch) \
                #             + ' lr=%.4e' % lr \
                #             + ' loss=%.4f total_loss=%.4f' % (loss, (sum_loss / (idx + 1)))
                print_str = 'Epoch {}/{}'.format(epoch, config.nepochs) \
                            + ' Modals: {}'.format(config.modals) \
                            + ' Iter {}/{}:'.format(idx + 1, config.niters_per_epoch)
                for key, value in METERS.items():
                    print_str += ' {}={:.3f} '.format(key, value.avg)
                del loss
                pbar.set_description(print_str, refresh=False)
            
            if (engine.distributed and (engine.local_rank == 0)) or (not engine.distributed):
                tb.add_scalar('train_loss', sum_loss / len(pbar), epoch)

            if (epoch >= config.checkpoint_start_epoch) and (epoch % config.checkpoint_step == 0) or (epoch == config.nepochs):
                if engine.distributed and (engine.local_rank == 0):
                    engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)
                elif not engine.distributed:
                    engine.save_and_link_checkpoint(config.checkpoint_dir,
                                                    config.log_dir,
                                                    config.log_dir_link)
            
                '''Evalution'''
                from dataloader.dataloader import ValPre
                from eval import RGBXSegEvaluator
                from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices

                parser = argparse.ArgumentParser()
                all_dev = parse_devices(args.devices)

                # network = dualsegmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
                data_setting = {'rgb_root': config.rgb_root_folder,
                                'rgb_format': config.rgb_format,
                                'gt_root': config.gt_root_folder,
                                'gt_format': config.gt_format,
                                'transform_gt': config.gt_transform,
                                'x_root':config.x_root_folder,
                                'x_format': config.x_format,
                                'x_single_channel': config.x_is_single_channel,
                                'class_names': config.class_names,
                                'train_source': config.train_source,
                                'eval_source': config.eval_source,
                                'class_names': config.class_names,
                                'num_labeled': config.num_labeled,
                                }
                val_pre = ValPre()
                dataset = RGBX_Base(data_setting, 'val', val_pre)
                with torch.no_grad():
                    config.modals = 'RGB'
                    segmentor = RGBXSegEvaluator(config, dataset, config.num_classes, config.norm_mean,
                                            config.norm_std, model.l_to_ab,
                                            config.eval_scale_array, config.eval_flip,
                                            all_dev, args.verbose, config.save_path,
                                            args.show_image)
                    segmentor.run_current(model, config.val_log_file,
                                config.link_val_log_file, tb, epoch)
                    config.modals = 'Depth'
                    segmentor = RGBXSegEvaluator(config, dataset, config.num_classes, config.norm_mean,
                                            config.norm_std, model.ab_to_l,
                                            config.eval_scale_array, config.eval_flip,
                                            all_dev, args.verbose, config.save_path,
                                            args.show_image)
                    segmentor.run_current(model, config.val_log_file,
                                config.link_val_log_file, tb, epoch)
                    config.modals = 'RGBD'
                    segmentor = RGBXSegEvaluator(config, dataset, config.num_classes, config.norm_mean,
                                            config.norm_std, model.l_and_ab,
                                            config.eval_scale_array, config.eval_flip,
                                            all_dev, args.verbose, config.save_path,
                                            args.show_image)
                    segmentor.run_current(model, config.val_log_file,
                                config.link_val_log_file, tb, epoch)
            logger_outputs(METERS, tb, epoch)
