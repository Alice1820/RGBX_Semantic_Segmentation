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