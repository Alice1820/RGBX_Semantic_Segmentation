import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # if args.n_gpu > 0:
    try:
        torch.cuda.manual_seed_all(seed)
    except:
        pass

def random_choices(subjects_train100, num_subjects):
    indexes = random.choices(range(len(subjects_train100)), k=num_subjects)
    # indexes = random.sample(range(len(train_subjects)), k=num_server_subjects)
    subjects = [subjects_train100[i] for i in indexes]
    return subjects

def random_choices_rgbd(rgb_train100, dep_train100, labels_train100, num_clips):
    assert (len(rgb_train100) == len(dep_train100) == len(labels_train100))
    indexes = random.choices(range(len(rgb_train100)), k=num_clips)
    rgb_choice = [rgb_train100[i] for i in indexes]
    dep_choice = [dep_train100[i] for i in indexes]
    labels_choice = [labels_train100[i] for i in indexes]
    return rgb_choice, dep_choice, labels_choice

def interleave(x, size):
    s = list(x.shape)
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])

class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def meter_outputs(METERS, outputs):
    for key, value in outputs.items():
        if key != 'logits':
            try:
                METERS[key].update(value.item())
            except:
                METERS[key].update(value)

def logger_outputs(METERS, tb, epoch):
    for key, value in METERS.items():
        if key != 'logits':
            tb.add_scalar('train/{}'.format(key), value.avg, epoch)