import argparse
import gc
import importlib
import json
import os
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from accelerate import Accelerator
from monai.data import decollate_batch
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR, UNet
from monai.optimizers import Novograd
from monai.transforms import (Activations, Activationsd, AsDiscrete,
                              AsDiscreted, Compose, Invertd,
                              KeepLargestConnectedComponentd, LoadImage,
                              Transposed)
from monai.utils import set_determinism
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from metric import HausdorffScore
from utils import *


def run_train(
    model,
    train_dataloader,
    optimizer,
    scheduler,
    seg_loss_func,
    cfg,
    # writer,
    epoch,
    # step,
    # iteration,
    accelerate
):
    model.train()
    scaler = GradScaler()
    progress_bar = tqdm(train_dataloader, total=len(train_dataloader))
    # tr_it = iter(train_dataloader)
    dataset_size = 0
    running_loss = 0.0

    for batch in progress_bar:
        # iteration += 1
        # batch = next(tr_it)
        inputs, masks = (
            batch["image"], #.to(cfg.device),
            batch["mask"], #.to(cfg.device),
        )

        # step += cfg.batch_size

        if cfg.amp is True:
            with autocast():
                outputs = model(inputs)
                loss = seg_loss_func(outputs, masks)
        else:
            outputs = model(inputs)
            loss = seg_loss_func(outputs, masks)
        if cfg.amp is True:
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(optimizer)
            scaler.update()
        else:
            #loss.backward()
            accelerate.backward(loss)
            optimizer.step()

        optimizer.zero_grad()
        scheduler.step()

        running_loss += loss.item() * cfg.batch_size
        dataset_size += cfg.batch_size
        losses = running_loss / dataset_size
        progress_bar.set_description(
            f"loss: {losses:.4f} lr: {optimizer.param_groups[0]['lr']:.6f}"
        )
        # del batch, inputs, masks, outputs, loss
    print(f"Train loss: {losses:.4f}")
    # torch.cuda.empty_cache()


def run_eval(
    model, val_dataloader, post_pred, metric_function, seg_loss_func, cfg, epoch
):

    model.eval()

    dice_metric, hausdorff_metric = metric_function

    progress_bar = tqdm(range(len(val_dataloader)))
    val_it = iter(val_dataloader)
    with torch.no_grad():
        for itr in progress_bar:
            batch = next(val_it)
            val_inputs, val_masks = (
                batch["image"], #.to(cfg.device),
                batch["mask"] #.to(cfg.device),
            )
            if cfg.val_amp is True:
                with autocast():
                    val_outputs = sliding_window_inference(
                        val_inputs, cfg.roi_size, cfg.sw_batch_size, model
                    )
            else:
                val_outputs = sliding_window_inference(
                    val_inputs, cfg.roi_size, cfg.sw_batch_size, model
                )
            # cal metric
            if cfg.run_tta_val is True:
                tta_ct = 1
                for dims in [[2], [3], [2, 3]]:
                    flip_val_outputs = sliding_window_inference(
                        torch.flip(val_inputs, dims=dims),
                        cfg.roi_size,
                        cfg.sw_batch_size,
                        model,
                    )
                    val_outputs += torch.flip(flip_val_outputs, dims=dims)
                    tta_ct += 1

                val_outputs /= tta_ct

            val_outputs = [post_pred(i) for i in val_outputs]
            val_outputs = torch.stack(val_outputs)
            # metric is slice level put (n, c, h, w, d) to (n, d, c, h, w) to (n*d, c, h, w)
            val_outputs = val_outputs.permute([0, 4, 1, 2, 3]).flatten(0, 1)
            val_masks = val_masks.permute([0, 4, 1, 2, 3]).flatten(0, 1)

            hausdorff_metric(y_pred=val_outputs, y=val_masks)
            dice_metric(y_pred=val_outputs, y=val_masks)

            del val_outputs, val_inputs, val_masks, batch

    dice_score = dice_metric.aggregate().item()
    hausdorff_score = hausdorff_metric.aggregate().item()
    dice_metric.reset()
    hausdorff_metric.reset()

    all_score = dice_score * 0.4 + hausdorff_score * 0.6
    print(
        f"dice_score: {dice_score} hausdorff_score: {hausdorff_score} all_score: {all_score}"
    )
    torch.cuda.empty_cache()

    return all_score
