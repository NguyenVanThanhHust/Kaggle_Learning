#!/usr/bin/env python
# coding: utf-8

# <center>
#     <h1>[Training] - FastAI Baseline</h1>
# <center>

# <center>
# <img src="https://hubmapconsortium.org/wp-content/uploads/2019/01/HuBMAP-Retina-Logo-Color.png">
# </center>

# # Description 
# 
# Welcome to Human BioMolecular Atlas Program (HuBMAP) + Human Protein Atlas (HPA) competition. 
# The objective of this challenge is segmentation of functional tissue units (FTU. e.g., glomeruli in kidney or alveoli in the lung) in biopsy slides from several different organs. 
# The underlying data includes imagery from different sources prepared with different protocols at a variety of resolutions, reflecting typical challenges for working with medical data.
# 
# This notebook provides a fast.ai starter Pytorch code based on a U-shape network (UneXt50) that was used on multiple competitions in the past and includes several tricks from the previous segmentation competitions.
# It is [dividing the images into tiles](https://www.kaggle.com/code/thedevastator/converting-to-256x256), selection of tiles with tissue, evaluation of the predictions of multiple models with TTA, combining the tile masks back into image level masks, and conversion into RLE. The [inference](https://www.kaggle.com/code/thedevastator/inference-fastai-baseline) is performed based on models trained in the [fast.ai training notebook](https://www.kaggle.com/code/thedevastator/training-fastai-baseline).
# 
# **Inference & Dataset Creation**
# 
# - #### Inference Notebook [here](https://www.kaggle.com/code/thedevastator/inference-fastai-baseline). 
# - #### Dataset Creation [here](https://www.kaggle.com/code/thedevastator/converting-to-256x256). 
# 
# **Precomputed Datasets**
# 
# - ##### [Dataset (512 x 512)](https://www.kaggle.com/datasets/thedevastator/hubmap-2022-512x512/)
# 
# - ##### [Dataset (256 x 256)](https://www.kaggle.com/datasets/thedevastator/hubmap-2022-256x256/)
# 
# - ##### [Dataset (128 x 128)](https://www.kaggle.com/datasets/thedevastator/hubmap-2022-128x128/settings)
# 
# ____
# 
# #### Everything is based on the excellent [notebooks](https://www.kaggle.com/code/iafoss/hubmap-pytorch-fast-ai-starter) by [iafoss](https://www.kaggle.com/iafoss) 
# All credit to belongs to the original author!
# ____

# In[1]:

from loguru import logger
try:
    get_ipython().run_line_magic('reload_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
    get_ipython().run_line_magic('matplotlib', 'inline')
except Exception as e:
    logger.info(e)
    pass 

import torch.nn as nn
from fastai.vision.all import PixelShuffle_ICNR, ConvLayer, Tensor, Metric, flatten_check
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import cv2
import gc
import random
from albumentations import *
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from os.path import isdir, isfile, join
from functools import wraps
import time
import torch
scaler = torch.cuda.amp.GradScaler()




from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


# In[2]:


SEED = 2021
TRAIN = '../input/hubmap-2022-256x256/train/'
MASKS = '../input/hubmap-2022-256x256/masks/'
LABELS = '../input/hubmap-organ-segmentation/train.csv'
SAVE_FILE = "../working/baseline_model"
EPOCHS = 300
IS_LOCAL = True
ACCUMULATION_STEPS = 1
NUM_CLASSES = 9
IS_LOCAL = True
IS_FULL = False
CLASSES = ['lung', 'largeintestine', 'prostate', 'kidney', 'spleen']

train_ids = [10044, 10274, 10666, 10912, 10971, 1184, 12233, 12244, 1229, 13483, 13942, 14396, 14407, 1500, 15706, 15732, 16149, 16609, 16659, 1690, 17143, 17187, 17455, 17828, 18422, 19084, 1955, 19569, 20247, 20428, 20955, 21086, 21155, 2174, 22016, 22059, 22995, 23009, 23640, 23828, 23959, 23961, 24194, 24269, 24961, 25430, 26982, 27471, 28318, 28622, 29213, 29223, 29296, 29307, 29809, 30080, 30294, 30355, 30414, 30424, 30765, 31898, 31958, 32009, 32126, 32412, 32741, 3409, 435, 4639, 4658, 4802, 4944, 5287, 5317, 5785, 5932, 5995, 6120, 10392, 10610, 10703, 10992, 1123, 11448, 11645, 12026, 12466, 12483, 144, 15551, 18792, 19179, 19360, 19377, 19507, 19997, 2079, 20831, 21358, 22236, 2279, 22953, 24833, 25472, 26664, 27781, 27803, 28126, 28657, 28748, 28963, 29143, 29690, 30201, 3054, 3057, 30581, 31290, 31675, 31733, 32231, 3959, 4404, 10488, 11064, 11629, 1220, 12452, 12476, 127, 12827, 13189, 14388, 15067, 15124, 15329, 16564, 1731, 1878, 20563, 23252, 24782, 25516, 25945, 26480, 27232, 2793, 28052, 28189, 28429, 29610, 30084, 30394, 30500, 31139, 31571, 31800, 32151, 4301, 4412, 4776, 5086, 5552, 10611, 11497, 1157, 12784, 13034, 13260, 14756, 15005, 15192, 15787, 15860, 16163, 16214, 16216, 16362, 164, 16711, 17422, 18121, 18401, 18426, 18449, 18777, 19048, 19533, 20302, 20440, 20478, 20520, 20794, 21021, 21112, 21129, 21195, 21321, 22035, 22133, 22544, 22718, 22741, 23051, 23243, 2344, 23665, 23760, 23880, 24097, 24100, 24222, 2424, 24241, 2447, 24522, 2500, 25620, 26101, 26174, 2668, 27298, 27350, 27468, 27616, 27879, 28262, 28436, 2874, 28823, 28940, 29238, 2943, 30194, 30224, 30250, 30474, 3083, 30876, 31698, 31709, 32325, 32527, 4066, 4265, 4777, 5099, 10651, 10892, 11662, 1168, 11890, 12174, 12471, 13396, 13507, 14183, 14674, 15499, 15842, 16728, 16890, 17126, 18445, 1850, 18900, 203, 21039, 21501, 21812, 22310, 23094, 25298, 25641, 25689, 26319, 26780, 26886, 2696, 27128, 27340, 27587, 27861, 28045, 28791, 29180, 29424, 29820, 31406, 31727, 31799, 3303, 351, 4062, 4561, 5832]
val_ids = [6390, 6730, 6794, 737, 7397, 7706, 7902, 8227, 8388, 8638, 8842, 9231, 928, 9358, 5583, 6722, 7169, 8116, 8876, 8894, 9407, 9453, 5777, 686, 7359, 8151, 8231, 8343, 9387, 9450, 5102, 6021, 62, 6318, 660, 6807, 7569, 7970, 8502, 9437, 9445, 9470, 9517, 9769, 9791, 6121, 6611, 676, 8222, 8402, 8450, 8752, 9777, 9904]     

if not isdir(TRAIN):
    TRAIN = '../../hubmap-organ-segmentation/hubmap-2022-256x256/image/'
    MASKS = '../../hubmap-organ-segmentation/hubmap-2022-256x256/mask/'
    LABELS = '../../hubmap-organ-segmentation/train.csv'
    SAVE_FILE = "cls_model"
    BATCH_SIZE=8
    IS_LOCAL = True
else:
    BATCH_SIZE=32
    train_ids = [10044, 10274, 10666, 10912, 10971, 1184, 12233, 12244, 1229, 13483, 13942, 14396, 14407, 1500, 15706, 15732, 16149, 16609, 16659, 1690, 17143, 17187, 17455, 17828, 18422, 19084, 1955, 19569, 20247, 20428, 20955, 21086, 21155, 2174, 22016, 22059, 22995, 23009, 23640, 23828, 23959, 23961, 24194, 24269, 24961, 25430, 26982, 27471, 28318, 28622, 29213, 29223, 29296, 29307, 29809, 30080, 30294, 30355, 30414, 30424, 30765, 31898, 31958, 32009, 32126, 32412, 32741, 3409, 435, 4639, 4658, 4802, 4944, 5287, 5317, 5785, 5932, 5995, 6120, 10392, 10610, 10703, 10992, 1123, 11448, 11645, 12026, 12466, 12483, 144, 15551, 18792, 19179, 19360, 19377, 19507, 19997, 2079, 20831, 21358, 22236, 2279, 22953, 24833, 25472, 26664, 27781, 27803, 28126, 28657, 28748, 28963, 29143, 29690, 30201, 3054, 3057, 30581, 31290, 31675, 31733, 32231, 3959, 4404, 10488, 11064, 11629, 1220, 12452, 12476, 127, 12827, 13189, 14388, 15067, 15124, 15329, 16564, 1731, 1878, 20563, 23252, 24782, 25516, 25945, 26480, 27232, 2793, 28052, 28189, 28429, 29610, 30084, 30394, 30500, 31139, 31571, 31800, 32151, 4301, 4412, 4776, 5086, 5552, 10611, 11497, 1157, 12784, 13034, 13260, 14756, 15005, 15192, 15787, 15860, 16163, 16214, 16216, 16362, 164, 16711, 17422, 18121, 18401, 18426, 18449, 18777, 19048, 19533, 20302, 20440, 20478, 20520, 20794, 21021, 21112, 21129, 21195, 21321, 22035, 22133, 22544, 22718, 22741, 23051, 23243, 2344, 23665, 23760, 23880, 24097, 24100, 24222, 2424, 24241, 2447, 24522, 2500, 25620, 26101, 26174, 2668, 27298, 27350, 27468, 27616, 27879, 28262, 28436, 2874, 28823, 28940, 29238, 2943, 30194, 30224, 30250, 30474, 3083, 30876, 31698, 31709, 32325, 32527, 4066, 4265, 4777, 5099, 10651, 10892, 11662, 1168, 11890, 12174, 12471, 13396, 13507, 14183, 14674, 15499, 15842, 16728, 16890, 17126, 18445, 1850, 18900, 203, 21039, 21501, 21812, 22310, 23094, 25298, 25641, 25689, 26319, 26780, 26886, 2696, 27128, 27340, 27587, 27861, 28045, 28791, 29180, 29424, 29820, 31406, 31727, 31799, 3303, 351, 4062, 4561, 5832]
    val_ids = [6390, 6730, 6794, 737, 7397, 7706, 7902, 8227, 8388, 8638, 8842, 9231, 928, 9358, 5583, 6722, 7169, 8116, 8876, 8894, 9407, 9453, 5777, 686, 7359, 8151, 8231, 8343, 9387, 9450, 5102, 6021, 62, 6318, 660, 6807, 7569, 7970, 8502, 9437, 9445, 9470, 9517, 9769, 9791, 6121, 6611, 676, 8222, 8402, 8450, 8752, 9777, 9904]     
    train_ids = train_ids + val_ids
    IS_LOCAL = False
    
if IS_FULL:
    train_ids, val_ids = train_ids + val_ids, train_ids + val_ids
    
    
NUM_WORKERS = 0

logger.info("{} {}".format(TRAIN, isdir(TRAIN)))
logger.info("NUM_TRAIN, NUM_VAL: {} {}".format(len(train_ids), len(val_ids)))
logger.info("BATCH_SIZE: {}".format(BATCH_SIZE))
logger.info("IS_LOCAL {}".format(IS_LOCAL))
logger.info("IS_FULL {}".format(IS_FULL))
logger.info("Accumulation step: {}".format(ACCUMULATION_STEPS))


# In[3]:


"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard




def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / union
        ious.append(iou)
    iou = f_mean(ious)    # mean accross images if per_image
    return 100 * iou


def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / union)
        ious.append(iou)
    ious = map(f_mean, zip(*ious)) # mean accross images if per_image
    return 100 * np.array(ious)


# --------------------------- BINARY LOSSES ---------------------------


def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = f_mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss


def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    #loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    loss = torch.dot(F.elu(errors_sorted)+1, Variable(grad))
    return loss


def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels


class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()


def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss


# --------------------------- MULTICLASS LOSSES ---------------------------


def lovasz_softmax(probas, labels, only_present=False, per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    if per_image:
        loss = f_mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), only_present=only_present)
                          for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), only_present=only_present)
    return loss


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (Variable(fg) - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return f_mean(losses)


def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """
    B, C, H, W = probas.size()
    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    labels = labels.view(-1)
    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels

def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)


# --------------------------- HELPER FUNCTIONS ---------------------------

def f_mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


# In[4]:



def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    #the following line gives ~10% speedup
    #but may lead to some stochasticity in the results 
    torch.backends.cudnn.benchmark = True
    
seed_everything(SEED)


# # Data
# One important thing here is the train/val split. To avoid possible leaks resulted by a similarity of tiles from the same images, it is better to keep tiles from each image together in train or in test.

# In[6]:


# https://www.kaggle.com/datasets/thedevastator/hubmap-2022-256x256
mean = np.array([0.7720342, 0.74582646, 0.76392896])
std = np.array([0.24745085, 0.26182273, 0.25782376])

def img2tensor(img,dtype:np.dtype=np.float32):
    if img.ndim==2 : img = np.expand_dims(img,2)
    img = np.transpose(img,(2,0,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class HuBMAPDataset(Dataset):
    def __init__(self, train=True, tfms=None):
        if train:
            ids = train_ids
        else:
            ids = val_ids

        all_labels = pd.read_csv(LABELS) 
        all_ids = all_labels.id.astype(int).values.tolist()
        all_organs = all_labels.organ.astype(str).values.tolist()
        self.cls = []
        self.fnames = []
        for fname in os.listdir(TRAIN):
            if fname.split('_')[0] in ids or int(fname.split('_')[0]) in ids:
                self.fnames.append(fname)
                index = all_ids.index(int(fname.split('_')[0]))
                label = all_organs[index]
                cls_index = CLASSES.index(label)
                self.cls.append(cls_index)

        self.train = train
        self.tfms = tfms
            
        logger.info("number of files {}".format(len(self.fnames)))

    def __len__(self):
        return len(self.fnames)
    
    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(MASKS,fname),cv2.IMREAD_GRAYSCALE)
        cls_index = self.cls[idx]
        if self.tfms is not None:
            augmented = self.tfms(image=img,mask=mask)
            img,mask = augmented['image'],augmented['mask']
        return img2tensor((img/255.0 - mean)/std), img2tensor(mask), torch.from_numpy(np.array(cls_index)).type(torch.LongTensor) 
    
def get_aug(p=1.0):
    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                         border_mode=cv2.BORDER_REFLECT),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        OneOf([
            HueSaturationValue(10,15,10),
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(),            
        ], p=0.3),
    ], p=p)


# # Model
# The model used in this kernel is based on a U-shape network (UneXt50, see image below), which I used in Severstal and Understanding Clouds competitions. The idea of a U-shape network is coming from a [Unet](https://arxiv.org/pdf/1505.04597.pdf) architecture proposed in 2015 for medical images: the encoder part creates a representation of features at different levels, while the decoder combines the features and generates a prediction as a segmentation mask. The skip connections between encoder and decoder allow us to utilize features from the intermediate conv layers of the encoder effectively, without a need for the information to go the full way through entire encoder and decoder. The latter is especially important to link the predicted mask to the specific pixels of the detected object. Later people realized that ImageNet pretrained computer vision models could drastically improve the quality of a segmentation model because of optimized architecture of the encoder, high encoder capacity (in contrast to one used in the original Unet), and the power of the transfer learning.
# 
# There are several important things that must be added to a Unet network, however, to make it able to reach competitive results with current state of the art approaches. First, it is **Feature Pyramid Network (FPN)**: additional skip connection between different upscaling blocks of the decoder and the output layer. So, the final prediction is produced based on the concatenation of U-net output with resized outputs of the intermediate layers. These skip-connections provide a shortcut for gradient flow improving model performance and convergence speed. Since intermediate layers have many channels, their upscaling and use as an input for the final layer would introduce a significant overhead in terms of the computational time and memory. Therefore, 3x3+3x3 convolutions are applied (factorization) before the resize to reduce the number of channels.
# 
# Another very important thing is the **Atrous Spatial Pyramid Pooling (ASPP) block** added between encoder and decoder. The flaw of the traditional U-shape networks is resulted by a small receptive field. Therefore, if a model needs to make a decision about a segmentation of a large object, especially for a large image resolution, it can get confused being able to look only into parts of the object. A way to increase the receptive field and enable interactions between different parts of the image is use of a block combining convolutions with different dilatations ([Atrous convolutions](https://arxiv.org/pdf/1606.00915.pdf) with various rates in ASPP block). While the original paper uses 6,12,18 rates, they may be customized for a particular task and a particular image resolution to maximize the performance. One more thing I added is using group convolutions in ASPP block to reduce the number of model parameters.
# 
# Finally, the decoder upscaling blocks are based on [pixel shuffle](https://arxiv.org/pdf/1609.05158.pdf) rather than transposed convolution used in the first Unet models. It allows to avoid artifacts in the produced masks. And I use [semisupervised Imagenet pretrained ResNeXt50](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models) model as a backbone. In Pytorch it provides the performance of EfficientNet B2-B3 with much faster convergence for the computational cost and GPU RAM requirements of EfficientNet B0 (though, in TF EfficientNet is highly optimized and may be a good thing to use).

# ![](https://i.ibb.co/z5KxDzm/Une-Xt50-1.png)

# In[7]:


class FPN(nn.Module):
    def __init__(self, input_channels:list, output_channels:list):
        super().__init__()
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv2d(in_ch, out_ch*2, kernel_size=3, padding=1),
             nn.ReLU(inplace=True), nn.BatchNorm2d(out_ch*2),
             nn.Conv2d(out_ch*2, out_ch, kernel_size=3, padding=1))
            for in_ch, out_ch in zip(input_channels, output_channels)])
        
    def forward(self, xs:list, last_layer):
        hcs = [F.interpolate(c(x),scale_factor=2**(len(self.convs)-i),mode='bilinear') 
               for i,(c,x) in enumerate(zip(self.convs, xs))]

        hcs.append(last_layer)
        return torch.cat(hcs, dim=1)

class UnetBlock(nn.Module):
    def __init__(self, up_in_c:int, x_in_c:int, nf:int=None, blur:bool=False,
                 self_attention:bool=False, **kwargs):
        super().__init__()
        self.shuf = PixelShuffle_ICNR(up_in_c, up_in_c//2, blur=blur, **kwargs)
        self.bn = nn.BatchNorm2d(x_in_c)
        ni = up_in_c//2 + x_in_c
        nf = nf if nf is not None else max(up_in_c//2,32)
        self.conv1 = ConvLayer(ni, nf, norm_type=None, **kwargs)
        self.conv2 = ConvLayer(nf, nf, norm_type=None,
            xtra=SelfAttention(nf) if self_attention else None, **kwargs)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, up_in:Tensor, left_in:Tensor) -> Tensor:
        s = left_in
        up_out = self.shuf(up_in)
        cat_x = self.relu(torch.cat([up_out, self.bn(s)], dim=1))
        return self.conv2(self.conv1(cat_x))
        
class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, groups=1):
        super().__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                stride=1, padding=padding, dilation=dilation, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class ASPP(nn.Module):
    def __init__(self, inplanes=512, mid_c=256, dilations=[6, 12, 18, 24], out_c=None):
        super().__init__()
        self.aspps = [_ASPPModule(inplanes, mid_c, 1, padding=0, dilation=1)] + \
            [_ASPPModule(inplanes, mid_c, 3, padding=d, dilation=d,groups=4) for d in dilations]
        self.aspps = nn.ModuleList(self.aspps)
        self.global_pool = nn.Sequential(nn.AdaptiveMaxPool2d((1, 1)),
                        nn.Conv2d(inplanes, mid_c, 1, stride=1, bias=False),
                        nn.BatchNorm2d(mid_c), nn.ReLU())
        out_c = out_c if out_c is not None else mid_c
        self.out_conv = nn.Sequential(nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False),
                                    nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(mid_c*(2+len(dilations)), out_c, 1, bias=False)
        self._init_weight()

    def forward(self, x):
        x0 = self.global_pool(x)
        xs = [aspp(x) for aspp in self.aspps]
        x0 = F.interpolate(x0, size=xs[0].size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x0] + xs, dim=1)
        return self.out_conv(x)
    
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


# In[8]:


class UneXt50(nn.Module):
    def __init__(self, stride=1, **kwargs):
        super().__init__()
        #encoder
        m = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models',
                           'resnext50_32x4d_ssl')
        self.enc0 = nn.Sequential(m.conv1, m.bn1, nn.ReLU(inplace=True))
        self.enc1 = nn.Sequential(nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1),
                            m.layer1) #256
        self.enc2 = m.layer2 #512
        self.enc3 = m.layer3 #1024
        self.enc4 = m.layer4 #2048
        #aspp with customized dilatations
        self.aspp = ASPP(2048,256,out_c=512,dilations=[stride*1,stride*2,stride*3,stride*4])
        self.drop_aspp = nn.Dropout2d(0.5)
        #decoder
        self.dec4 = UnetBlock(512,1024,256)
        self.dec3 = UnetBlock(256,512,128)
        self.dec2 = UnetBlock(128,256,64)
        self.dec1 = UnetBlock(64,64,32)
        self.fpn = FPN([512,256,128,64],[16]*4)
        
        self.pool_layer = nn.MaxPool2d(64, stride=4)
        self.cls_layer = nn.Linear(27744, NUM_CLASSES)


        self.drop = nn.Dropout2d(0.1)
        self.final_conv = ConvLayer(32+16*4, 1, ks=1, norm_type=None, act_cls=None)
        
    def forward(self, x):
        enc0 = self.enc0(x)
        enc1 = self.enc1(enc0)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.aspp(enc4)


        dec3 = self.dec4(self.drop_aspp(enc5),enc3)
        dec2 = self.dec3(dec3,enc2)
        dec1 = self.dec2(dec2,enc1)
        dec0 = self.dec1(dec1,enc0)

        x = self.fpn([enc5, dec3, dec2, dec1], dec0)
        cls_feature = self.pool_layer(x)
        cls_feature = cls_feature.view(cls_feature.size(0), -1)
        
        pred_cls = self.cls_layer(cls_feature)

        x = self.final_conv(self.drop(x))
        x = F.interpolate(x,scale_factor=2,mode='bilinear')
        return x, pred_cls

#split the model to encoder and decoder for fast.ai
split_layers = lambda m: [list(m.enc0.parameters())+list(m.enc1.parameters())+
                list(m.enc2.parameters())+list(m.enc3.parameters())+
                list(m.enc4.parameters()),
                list(m.aspp.parameters())+list(m.dec4.parameters())+
                list(m.dec3.parameters())+list(m.dec2.parameters())+
                list(m.dec1.parameters())+list(m.fpn.parameters())+
                list(m.final_conv.parameters())]


# # Loss and metric
# A famous loss for image segmentation is the [Lovász loss](https://arxiv.org/pdf/1705.08790.pdf), a surrogate of IoU. Following [iafoss](https://www.kaggle.com/iafoss)'s [work](https://www.kaggle.com/code/iafoss/hubmap-pytorch-fast-ai-starter):
# - **ReLU in it must be replaced by (ELU + 1)**(, like he did [here](https://www.kaggle.com/iafoss/lovasz).
# - **Symmetric Lovász loss:** consider not only a predicted segmentation and a provided mask but also the inverse prediction and the inverse mask (predict mask for negative case).

# In[9]:


def symmetric_lovasz(outputs, targets):
    return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))


# In[10]:


class Dice_soft(Metric):
    def __name__(self, ):
        return "Dice soft"
    
    def __init__(self, axis=1): 
        self.axis = axis 
        self.inter = 0.0
        self.union = 0
        
    def reset(self): self.inter,self.union = 0,0
    def accumulate(self, preds, gts):
        pred,targ = flatten_check(torch.sigmoid(preds), gts)
        self.inter += (pred*targ).float().sum().item()
        self.union += (pred+targ).float().sum().item()
    
    @property
    def value(self): return 2.0 * self.inter/self.union if self.union > 0 else None
    


# dice with automatic threshold selection
class Dice_th(Metric):
    def __name__(self, ):
        return "Dice th"
    
    def __init__(self, ths=np.arange(0.1,0.9,0.05), axis=1): 
        self.axis = axis
        self.ths = ths
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))
    
    def reset(self): 
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))
        
    def accumulate(self, preds, gts):
        pred,targ = flatten_check(torch.sigmoid(preds), gts)
        for i,th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p*targ).float().sum().item()
            self.union[i] += (p+targ).float().sum().item()

    @property
    def value(self):
        dices = torch.where(self.union > 0.0, 
                2.0*self.inter/self.union, torch.zeros_like(self.union))
        return dices.max()


from torchmetrics import Accuracy
class Acc():
    def __init__(self) -> None:
        self.acc = 0
        self.total = 0
        self.correct = 0
        self.value = 0

    def __name__(self, ):
        return "Accuracy"

    def accumulate(self, preds, tgts):
        _, pred_cls = preds.max(1)
        self.total += tgts.shape[0]
        self.correct += pred_cls.eq(tgts).sum().item()
        self.value = self.correct / self.total



# # Model evaluation

# In[11]:


def save_img(data,name,out):
    data = data.float().cpu().numpy()
    img = cv2.imencode('.png',(data*255).astype(np.uint8))[1]
    out.writestr(name, img)


# # Train

# In[12]:


try:
    import segmentation_models_pytorch as smp
    from segmentation_models_pytorch import utils as smp_utils
except Exception as e:
    try:
        get_ipython().system('pip install segmentation_models_pytorch')
        import segmentation_models_pytorch as smp
        from segmentation_models_pytorch import utils as smp_utils
    except Exception as e:
        print(e)


# In[13]:


class AverageValueMeter():
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan


# In[14]:


#@timeit
def train_one_epoch(model, dataloader, optimizer, loss_fn, metrics):
    logs = {}
    loss_meter = AverageValueMeter()
    metrics_meters = {metric.__name__(): AverageValueMeter() for metric in metrics}

    mask_loss_fn, cls_loss_fn = loss_fn
    for batch_idx, (imgs, masks, tgts) in enumerate(dataloader):
        with torch.cuda.amp.autocast():
            imgs, masks, tgts = imgs.cuda(), masks.cuda(), tgts.cuda()
            preds = model(imgs)
            mask_preds, cls_preds = preds

            mask_loss = mask_loss_fn(mask_preds, masks)
            cls_loss = cls_loss_fn(cls_preds, tgts)
            loss = cls_loss + mask_loss

            # loss.backward()
            scaler.scale(loss).backward()
            
            loss = loss / ACCUMULATION_STEPS
            
            # update loss logs
            loss_value = loss.cpu().detach().numpy()
            loss_meter.add(loss_value)
            loss_logs = {"loss": loss_meter.mean}
            logs.update(loss_logs)

            # weights update
            if ((batch_idx + 1) % ACCUMULATION_STEPS == 0) or (batch_idx + 1 == len(dataloader)):
                scaler.step(optimizer)

                # Updates the scale for next iteration
                scaler.update()
                optimizer.zero_grad()

            # update metrics logs
            for idx, metric_fn in enumerate(metrics):
                if idx == 0:
                    metric_fn.accumulate(cls_preds, tgts)
                    metric_value = metric_fn.value
                    metrics_meters[metric_fn.__name__()].add(metric_value)

                else:
                    metric_fn.accumulate(mask_preds, masks)
                    metric_value = metric_fn.value
                    metrics_meters[metric_fn.__name__()].add(metric_value)
            metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
            logs.update(metrics_logs)
    return logs


# In[16]:


#@timeit
def val_one_epoch(model, dataloader, optimizer, loss_fn, metrics):
    
    logs = {}
    loss_meter = AverageValueMeter()
    metrics_meters = {metric.__name__(): AverageValueMeter() for metric in metrics}
    model.eval()

    mask_loss_fn, cls_loss_fn = loss_fn
    for batch_idx, (imgs, masks, tgts) in enumerate(dataloader):
        imgs, masks, tgts = imgs.cuda(), masks.cuda(), tgts.cuda()
        preds = model(imgs)
        mask_preds, cls_preds = preds

        mask_loss = mask_loss_fn(mask_preds, masks)
        cls_loss = cls_loss_fn(cls_preds, tgts)
        loss = cls_loss + mask_loss
        
        # update loss logs
        loss_value = loss.cpu().detach().numpy()
        loss_meter.add(loss_value)
        loss_logs = {"loss": loss_meter.mean}

        logs.update(loss_logs)

        # update metrics logs
        for idx, metric_fn in enumerate(metrics):
            if idx == 0:
                metric_fn.accumulate(cls_preds, tgts)
                metric_value = metric_fn.value
                metrics_meters[metric_fn.__name__()].add(metric_value)

            else:
                metric_fn.accumulate(mask_preds, masks)
                metric_value = metric_fn.value
                metrics_meters[metric_fn.__name__()].add(metric_value)

        metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
        logs.update(metrics_logs)
    return logs


# In[17]:


model = UneXt50().cuda() 


# In[18]:




        
    
metrics=[Acc(), Dice_soft(),Dice_th()]
optimizer = torch.optim.SGD([ 
    dict(params=model.parameters(), lr=0.00001),
])

DEVICE = 'cuda'


# In[24]:


def train_local():
    previous_path = "nothing"
    max_score = 0.0
    cls_loss = nn.CrossEntropyLoss()
    for epoch in range(0, EPOCHS):
        logger.info('Epoch: {}'.format(epoch))
        train_logs = train_one_epoch(model, train_dataloader, optimizer, (symmetric_lovasz, cls_loss), metrics)
        valid_logs = val_one_epoch(model, val_dataloader, optimizer, (symmetric_lovasz, cls_loss), metrics)
        # do something (save model, change lr, etc.)
        if max_score < valid_logs['Dice soft']:
            max_score = valid_logs['Dice soft']
            model_info = "_{}_{}_{}.pth".format(epoch,BATCH_SIZE, round(max_score, 4))
            save_model = SAVE_FILE+model_info
            torch.save(model, save_model)
            logger.info('Model saved at {}'.format( save_model))
            if os.path.isfile(previous_path):
                os.remove(previous_path)
                logger.info("removed: {}".format(previous_path))
            previous_path = save_model
        if epoch == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            logger.info('Decrease decoder learning rate to 1e-5!')


# In[ ]:



# In[25]:


def train_online():
    previous_path = "nothing"
    for epoch in range(0, EPOCHS):
        logger.info('Epoch: {}'.format(epoch))
        train_logs = train_one_epoch(model, train_dataloader, optimizer, symmetric_lovasz, metrics)
        logger.info(train_logs)
        model_info = "_{}_{}.pth".format(epoch, BATCH_SIZE,)
        save_model = SAVE_FILE+model_info
        torch.save(model, save_model)
        logger.info('Model saved at {}'.format( save_model))
        if os.path.isfile(previous_path):
            os.remove(previous_path)
            logger.info("removed: {}".format(previous_path))
        previous_path = save_model


# In[26]:


dice = Dice_th(np.arange(0.2,0.7,0.1))

train_dataset = HuBMAPDataset(train=True, tfms=get_aug())
val_dataset = HuBMAPDataset(train=False)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS, drop_last=False, pin_memory=True, shuffle=True)

val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS, drop_last=False, pin_memory=True, shuffle=False)

from datetime import datetime

now = datetime.now()

if IS_LOCAL:
    logger.info("Train local", end=" ")
    if IS_FULL:
        logger.info("full")
    else:
        logger.info(" 80%")
    train_local()
else:
    logger.info("Train online")
    train_online()
now = datetime.now()



# In[30]:

