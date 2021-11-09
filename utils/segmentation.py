import torch
import torch.nn.functional as F
import glob
import matplotlib.pyplot as plt
import PIL
from utils.model import BiSeNet

import os.path as osp
import time
import sys
import logging

import torch.distributed as dist
import logging

def setup_logger(logpth):
    logfile = 'BiSeNet-{}.log'.format(time.strftime('%Y-%m-%d-%H-%M-%S'))
    logfile = osp.join(logpth, logfile)
    FORMAT = '%(levelname)s %(filename)s(%(lineno)d): %(message)s'
    log_level = logging.INFO
    if dist.is_initialized() and not dist.get_rank()==0:
        log_level = logging.ERROR
    logging.basicConfig(level=log_level, format=FORMAT, filename=logfile)
    logging.root.addHandler(logging.StreamHandler())

import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

def vis_parsing_maps(parsing_anno, stride):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    #vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.0, vis_parsing_anno_color, 1.0, 0)

    return vis_parsing_anno_color

"""def evaluate(respth='./res/test_res', dspth='./data', cp='model_final_diss.pth'):
    if not os.path.exists(respth):
        os.makedirs(respth)

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    save_pth = osp.join('/home/sayantan/seginstyle/face-parsing.PyTorch/res/cp', cp)
    net.load_state_dict(torch.load(save_pth))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        for image_path in os.listdir(dspth):
            img = Image.open(osp.join(dspth, image_path))
            image = img.resize((512, 512), Image.BILINEAR)
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            # print(parsing)
            # print(np.unique(parsing))

            vis_parsing_maps(image, parsing, stride=1, save_im=True, save_path=osp.join(respth, image_path))
"""
def evaluate_numpy(arr):

    n_classes = 19
    net = BiSeNet(n_classes=n_classes)
    net.cuda()
    net.load_state_dict(torch.load("/content/faceparse.pth"))
    net.eval()

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    with torch.no_grad():
        #img = Image.open(osp.join(dspth, image_path))
        #image = img.resize((512, 512), Image.BILINEAR)
        #img = to_tensor(image)
        #img = torch.unsqueeze(arr, 0)
        #img = img.cuda()
        out = net(arr)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
        # print(parsing)
        # print(np.unique(parsing))

        return vis_parsing_maps(parsing, stride=1)
        


def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.

    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch

    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat)
    B_sum = torch.sum(iflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

from torch.autograd import Variable
def segloss(generated_images,real_images):
    gen_mask  = evaluate_numpy(generated_images)
    print(gen_mask.shape)
    real_mask = evaluate_numpy(real_images)
    return dice_loss(gen_mask,real_mask)

#evaluate(respth='/home/sayantan/seginstyle/face-parsing.PyTorch/res/test_res',dspth='/home/sayantan/seginstyle/face-parsing.PyTorch/data/test-img', cp='79999_iter.pth')
