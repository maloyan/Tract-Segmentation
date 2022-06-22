import copy
import gc
import os
import random
import shutil
import time
from collections import defaultdict
from glob import glob

import albumentations as A
import cupy as cp
import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from albumentations.pytorch import ToTensorV2
from colorama import Back, Fore, Style
from IPython import display as ipd
from joblib import Parallel, delayed
from matplotlib.patches import Rectangle
from sklearn.model_selection import (KFold, StratifiedGroupKFold,
                                     StratifiedKFold)
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def set_seed(seed=42):
    """Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print("> SEEDING DONE")


def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    img = np.tile(img[..., None], [1, 1, 3])  # gray to rgb
    img = img.astype("float32")  # original is uint16
    mx = np.max(img)
    if mx:
        img /= mx  # scale image to [0, 1]
    return img


def load_msk(path):
    msk = np.load(path)
    msk = msk.astype("float32")
    msk /= 255.0
    return msk


def get_metadata(row):
    data = row["id"].split("_")
    case = int(data[0].replace("case", ""))
    day = int(data[1].replace("day", ""))
    slice_ = int(data[-1])
    row["case"] = case
    row["day"] = day
    row["slice"] = slice_
    return row


def path2info(row):
    path = row["image_path"]
    data = path.split("/")
    slice_ = int(data[-1].split("_")[1])
    case = int(data[-3].split("_")[0].replace("case", ""))
    day = int(data[-3].split("_")[1].replace("day", ""))
    width = int(data[-1].split("_")[2])
    height = int(data[-1].split("_")[3])
    row["height"] = height
    row["width"] = width
    row["case"] = case
    row["day"] = day
    row["slice"] = slice_
    #     row['id'] = f'case{case}_day{day}_slice_{slice_}'
    return row

def mask2rle(msk, thr=0.5):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    msk    = cp.array(msk)
    pixels = msk.flatten()
    pad    = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs   = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def masks2rles(msks, ids, heights, widths):
    pred_strings = []; pred_ids = []; pred_classes = [];
    for idx in range(msks.shape[0]):
        height = heights[idx].item()
        width = widths[idx].item()
        msk = cv2.resize(msks[idx], 
                         dsize=(width, height), 
                         interpolation=cv2.INTER_NEAREST) # back to original shape
        rle = [None]*3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk[...,midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]]*len(rle))
        pred_classes.extend(['large_bowel', 'small_bowel', 'stomach'])
    return pred_strings, pred_ids, pred_classes


# def id2mask(id_):
#     idf = df[df["id"] == id_]
#     wh = idf[["height", "width"]].iloc[0]
#     shape = (wh.height, wh.width, 3)
#     mask = np.zeros(shape, dtype=np.uint8)
#     for i, class_ in enumerate(["large_bowel", "small_bowel", "stomach"]):
#         cdf = idf[idf["class"] == class_]
#         rle = cdf.segmentation.squeeze()
#         if len(cdf) and not pd.isna(rle):
#             mask[..., i] = rle_decode(rle, shape[:2])
#     return mask


# def rgb2gray(mask):
#     pad_mask = np.pad(mask, pad_width=[(0, 0), (0, 0), (1, 0)])
#     gray_mask = pad_mask.argmax(-1)
#     return gray_mask


# def gray2rgb(mask):
#     rgb_mask = tf.keras.utils.to_categorical(mask, num_classes=4)
#     return rgb_mask[..., 1:].astype(mask.dtype)


# def show_img(img, mask=None):
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     #     img = clahe.apply(img)
#     #     plt.figure(figsize=(10,10))
#     plt.imshow(img, cmap="bone")

#     if mask is not None:
#         # plt.imshow(np.ma.masked_where(mask!=1, mask), alpha=0.5, cmap='autumn')
#         plt.imshow(mask, alpha=0.5)
#         handles = [
#             Rectangle((0, 0), 1, 1, color=_c)
#             for _c in [(0.667, 0.0, 0.0), (0.0, 0.667, 0.0), (0.0, 0.0, 0.667)]
#         ]
#         labels = ["Large Bowel", "Small Bowel", "Stomach"]
#         plt.legend(handles, labels)
#     plt.axis("off")


# # ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# def rle_decode(mask_rle, shape):
#     """
#     mask_rle: run-length as string formated (start length)
#     shape: (height,width) of array to return
#     Returns numpy array, 1 - mask, 0 - background

#     """
#     s = mask_rle.split()
#     starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
#     starts -= 1
#     ends = starts + lengths
#     img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
#     for lo, hi in zip(starts, ends):
#         img[lo:hi] = 1
#     return img.reshape(shape)  # Needed to align to RLE direction


# # ref.: https://www.kaggle.com/stainsby/fast-tested-rle
# def rle_encode(img):
#     """
#     img: numpy array, 1 - mask, 0 - background
#     Returns run length as string formated
#     """
#     pixels = img.flatten()
#     pixels = np.concatenate([[0], pixels, [0]])
#     runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
#     runs[1::2] -= runs[::2]
#     return " ".join(str(x) for x in runs)

