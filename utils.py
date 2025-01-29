import random
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import os
import logging



class AverageMeter:
    # Sourced from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    def __init__(self):
        self.reset()
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum*1.0 / self.count*1.0

def get_logger(folder):
    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    if not os.path.isdir(folder):
        os.mkdir(folder)
    fh = logging.FileHandler(os.path.join(folder, 'checkpoint.log'), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def get_accuracy(y_prob, y_trues, return_vec=False):
    '''
    Calculates the task and class incremental accuracy of the model
    '''
    # global acc_chosen
    y_pred = torch.argmax(y_prob, axis=1)
    # print('y_true:',y_trues)
    # print('y_pred:',y_pred)
    acc_full = torch.eq(y_pred, y_trues).float()
    if return_vec:
        return acc_full, y_pred

    return (acc_full*100.0).mean(), y_pred
def seed_everything(seed):
    '''
    Fixes the class-to-task assignments and most other sources of randomness, except CUDA training aspects.
    '''
    # Avoid all sorts of randomness for better replication
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True # An exemption for speed :P

def save_model(opt, model):
    '''
    Used for saving the pretrained model, not for intermediate breaks in running the code.
    '''
    state = {'opt': opt,
        'state_dict': model.state_dict()}
    filename = opt.log_dir+opt.old_exp_name+'/pretrained_model.pth.tar'
    torch.save(state, filename)


def load_model(opt, model, logger):
    '''
    Dynamically loads pretrained model from the previous task, handling mismatches if there is in fc3 class number.
    '''
    filepath = opt.log_dir + opt.old_exp_name + '/pretrained_model.pth.tar'
    assert os.path.isfile(filepath), f"Checkpoint not found at {filepath}"
    logger.debug("=> loading checkpoint '{}'".format(filepath))
    checkpoint = torch.load(filepath, map_location=torch.device('cuda'))

    model_state = model.state_dict()
    checkpoint_state = checkpoint['state_dict']

    # Handle fc3 if there is a size mismatch
    if 'fc3.weight' in checkpoint_state:
        saved_num_classes = checkpoint_state['fc3.weight'].size(0)
        current_num_classes = model_state['fc3.weight'].size(0)

        if saved_num_classes != current_num_classes:
            logger.info(f"Resizing fc3 layer from {saved_num_classes} to {current_num_classes}")
            model_state['fc3.weight'][:saved_num_classes] = checkpoint_state['fc3.weight']
            model_state['fc3.bias'][:saved_num_classes] = checkpoint_state['fc3.bias']

            if current_num_classes > saved_num_classes:
                nn.init.xavier_uniform_(model_state['fc3.weight'][saved_num_classes:])
                model_state['fc3.bias'][saved_num_classes:].fill_(0)
        else:
            model_state.update(checkpoint_state)

    model.load_state_dict(model_state)
    return model


def cutmix_data(x, y, alpha=1.0, cutmix_prob=0.5):
    assert(alpha > 0)
    # generate mixed sample
    lam = np.random.beta(alpha, alpha)

    batch_size = x.size()[0]
    index = torch.randperm(batch_size)

    if torch.cuda.is_available():
        index = index.cuda()

    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]

    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def create_class_mask(num_tasks, num_classes, current_task):
    class_mask = [0] * (num_tasks * num_classes)
    class_mask[:current_task * num_classes] = [1] * (current_task * num_classes)
    return class_mask

def create_class_mask_realistic(class_list, old_classes):
      old_class_mask = [1 if item in old_classes else 0 for item in class_list]
      return old_class_mask

def semirealistic_divide_classes(classes, num_segments):
    if classes < num_segments:
        raise ValueError("Number of tasks cannot be greater than number of classes")

    classes = list(range(classes))
    random.shuffle(classes)
    segments = []
    for i in range(num_segments):
        max_segment_size = len(classes) // (num_segments - i)
        segment_size = random.randint(1, max_segment_size)
        if len(classes) - segment_size < num_segments - i:
            segment_size = len(classes)
        segment_classes = random.sample(classes, segment_size)
        segments.append(segment_classes)
        classes = [cls for cls in classes if cls not in segment_classes]
    for i in range(len(classes)):
        segments[i % num_segments].append(classes[i])
    return segments