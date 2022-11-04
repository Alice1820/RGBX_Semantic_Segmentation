from json import load
import os
import cv2
import argparse
import numpy as np

import PIL.Image as Image
import torch
import torch.nn as nn

# from config import config
from utils.pyt_utils import ensure_dir, link_file, load_model, parse_devices
from utils.visualize import print_iou, show_img
from engine.evaluator import Evaluator
from engine.logger import get_logger
from utils.metric import hist_info, compute_score
from dataloader.RGBXDataset import RGBX_X, RGBX_Base
from models.dual_builder import RGBXEncoderDecoder as dualsegmodel
from models.builder import EncoderDecoder as segmodel
from models.create_model import *
from dataloader.dataloader import ValPre
from utils.pyt_utils import all_reduce_tensor, extant_file, load_model
from importlib import import_module


logger = get_logger()

class RGBXSegEvaluator(Evaluator):
    def func_per_iteration(self, data, device):
        img = data['data']
        label = data['label']
        modal_x = data['modal_x']
        name = data['fn']
        if self.config.modals == 'RGBD':
            pred = self.sliding_eval_rgbX(img, modal_x, self.config.eval_crop_size, self.config.eval_stride_rate, device)
        elif self.config.modals == 'RGB':
            pred = self.sliding_eval(img, self.config.eval_crop_size, self.config.eval_stride_rate, device)
        elif self.config.modals == 'Depth':
            pred = self.sliding_eval(modal_x, self.config.eval_crop_size, self.config.eval_stride_rate, device)

        hist_tmp, labeled_tmp, correct_tmp = hist_info(self.config.num_classes, pred, label)
        results_dict = {'hist': hist_tmp, 'labeled': labeled_tmp, 'correct': correct_tmp}

        if self.save_path is not None:
            ensure_dir(os.path.join(self.save_path))
            ensure_dir(os.path.join(self.save_path, 'results'))
            ensure_dir(os.path.join(self.save_path, 'results_color'))

            fn = name + '.png'

            # save colored result
            result_img = Image.fromarray(pred.astype(np.uint8), mode='P')
            class_colors = self.dataset.get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path, 'results_color', fn))

            # save raw result
            cv2.imwrite(os.path.join(self.save_path, 'results', fn), pred)
            # logger.info('Save the image ' + fn)

            '''Ground Truth'''
            ensure_dir(os.path.join(self.save_path, 'gts'))
            ensure_dir(os.path.join(self.save_path, 'gts_color'))

            result_img = Image.fromarray(label.astype(np.uint8), mode='P')
            class_colors = self.dataset.get_class_colors()
            palette_list = list(np.array(class_colors).flat)
            if len(palette_list) < 768:
                palette_list += [0] * (768 - len(palette_list))
            result_img.putpalette(palette_list)
            result_img.save(os.path.join(self.save_path, 'gts_color', fn))

            # save raw result
            cv2.imwrite(os.path.join(self.save_path, 'gts', fn), label)

        if self.show_image:
            colors = self.dataset.get_class_colors
            image = img
            clean = np.zeros(label.shape)
            comp_img = show_img(colors, config.background, image, clean,
                                label,
                                pred)
            cv2.imshow('comp_image', comp_img)
            cv2.waitKey(0)

        return results_dict

    def compute_metric(self, results):
        hist = np.zeros((self.config.num_classes, self.config.num_classes))
        correct = 0
        labeled = 0
        count = 0
        for d in results:
            hist += d['hist']
            correct += d['correct']
            labeled += d['labeled']
            count += 1

        iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc = compute_score(hist, correct, labeled)
        result_line = print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
                                self.dataset.class_names, show_no_back=False)
        # log_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc,
        #                         self.dataset.class_names, show_no_back=False)
        return iou, mean_IoU, _, freq_IoU, mean_pixel_acc, pixel_acc, result_line

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', default='last', type=str)
    parser.add_argument('-d', '--devices', default='0', type=str)
    parser.add_argument('-v', '--verbose', default=False, action='store_true')
    parser.add_argument('--show_image', '-s', default=False,
                        action='store_true')
    parser.add_argument('--save_path', '-p', default='/data0/xfzhang/data/NYUv2/results/')
    parser.add_argument('--config', type=str, default=None)
    # parser.add_argument('--continue', type=extant_file,
    parser.add_argument('--continue', type=str,
                    # metavar="FILE",
                    dest="continue_fpath",
                    help='continue from one certain checkpoint')
    args = parser.parse_args()
    
    config = import_module(args.config)
    # print (config)
    # from config import config
    config = config.config

    all_dev = parse_devices(args.devices)

    # network = dualsegmodel(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
    model = create_model(config=config, criterion=None, norm_layer=nn.BatchNorm2d)
    tmp = torch.load(args.continue_fpath, map_location=torch.device('cpu'))['model']
    load_model(model, tmp)
    logger.info("Loaded checkpoint from {}.".format(args.continue_fpath))
    test_model = model
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
        if config.modals == 'RGBD':
            config.modals = 'RGB'
            segmentor = RGBXSegEvaluator(config, dataset, config.num_classes, config.norm_mean,
                                    config.norm_std, test_model.l_to_ab,
                                    config.eval_scale_array, config.eval_flip,
                                    all_dev, args.verbose, config.save_path,
                                    args.show_image)
            segmentor.run_current(test_model, config.val_log_file,
                        config.link_val_log_file)
            config.modals = 'Depth'
            segmentor = RGBXSegEvaluator(config, dataset, config.num_classes, config.norm_mean,
                                    config.norm_std, test_model.ab_to_l,
                                    config.eval_scale_array, config.eval_flip,
                                    all_dev, args.verbose, config.save_path,
                                    args.show_image)
            segmentor.run_current(test_model, config.val_log_file,
                        config.link_val_log_file)
            config.modals = 'RGBD'
            segmentor = RGBXSegEvaluator(config, dataset, config.num_classes, config.norm_mean,
                                    config.norm_std, test_model.l_and_ab,
                                    config.eval_scale_array, config.eval_flip,
                                    all_dev, args.verbose, config.save_path,
                                    args.show_image)
            segmentor.run_current(test_model, config.val_log_file,
                        config.link_val_log_file)
        else:
            segmentor = RGBXSegEvaluator(config, dataset, config.num_classes, config.norm_mean,
                                    config.norm_std, test_model,
                                    config.eval_scale_array, config.eval_flip,
                                    all_dev, args.verbose, config.save_path,
                                    args.show_image)
            segmentor.run_current(test_model, config.val_log_file,
                        config.link_val_log_file)