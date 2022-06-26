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

from engine import run_eval, run_train
from metric import HausdorffScore
from utils import *

sys.path.append("configs")

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "-c", "--config", default="cfg_unet_multilabel", help="config filename"
)
parser.add_argument("-f", "--fold", type=int, default=0, help="fold")
parser.add_argument("-s", "--seed", type=int, default=20220421, help="seed")
parser.add_argument("-w", "--weights", default=None, help="the path of weights")

parser_args, _ = parser.parse_known_args(sys.argv)

cfg = importlib.import_module(parser_args.config).cfg
cfg.fold = parser_args.fold
cfg.seed = parser_args.seed
cfg.weights = parser_args.weights


accelerate = Accelerator()

cfg.data_json_dir = cfg.data_dir + f"dataset_3d_fold_{cfg.fold}.json"

with open(cfg.data_json_dir, "r", encoding="utf-8") as f:
    cfg.data_json = json.load(f)

os.makedirs(str(cfg.output_dir + f"/fold{cfg.fold}/"), exist_ok=True)


train_dataset = get_train_dataset(cfg)
train_dataloader = get_train_dataloader(train_dataset, cfg)

val_dataset = get_val_dataset(cfg)
val_dataloader = get_val_dataloader(val_dataset, cfg)

print(f"run fold {cfg.fold}, train len: {len(train_dataset)}")

model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=3,
    channels=(32, 64, 128, 256, 512),
    strides=(2, 2, 2, 2),
    kernel_size=3,
    up_kernel_size=3,
    num_res_units=2,
    act="PRELU",
    norm="BATCH",
    dropout=0.2,
    bias=True,
    dimensions=None,
) #.to(cfg.device)

# model = SwinUNETR(
#     in_channels=1,
#     out_channels=3,
#     img_size=cfg.img_size,
#     feature_size=48,
#     use_checkpoint=True,
# )

# weight = torch.load(
#     os.path.join("/root/tract_segmentation/checkpoints/model_swinvit.pt")
# )

# model.load_from(weights=weight)
# model.to(cfg.device)

if cfg.weights is not None:
    model.load_state_dict(
        torch.load(os.path.join(f"{cfg.output_dir}/fold{cfg.fold}", cfg.weights))[
            "model"
        ]
    )
    print(f"weights from: {cfg.weights} are loaded.")

total_steps = len(train_dataset)
optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

if cfg.lr_mode == "cosine":
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs * (total_steps // cfg.batch_size),
        eta_min=cfg.min_lr,
    )

elif cfg.lr_mode == "warmup_restart":
    scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=cfg.restart_epoch * (total_steps // cfg.batch_size),
        T_mult=1,
        eta_min=cfg.min_lr,
    )


seg_loss_func = DiceBceMultilabelLoss()
metric_function = [DiceMetric(reduction="mean"), HausdorffScore(reduction="mean")]

post_pred = Compose(
    [
        Activations(sigmoid=True),
        AsDiscrete(threshold=0.5),
    ]
)
model, optimizer, train_dataloader, val_dataloader= accelerate.prepare(
    model, optimizer, train_dataloader, val_dataloader
)


# train and val loop
# step = 0
best_val_metric = 0.0

for epoch in range(cfg.epochs):
    print("EPOCH:", epoch)
    run_train(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        seg_loss_func=seg_loss_func,
        cfg=cfg,
        # writer=writer,
        epoch=epoch,
        # step=step,
        # iteration=i,
        accelerate=accelerate,
    )

    if epoch % cfg.eval_epochs == 0:
        val_metric = run_eval(
            model=model,
            val_dataloader=val_dataloader,
            post_pred=post_pred,
            metric_function=metric_function,
            seg_loss_func=seg_loss_func,
            cfg=cfg,
            epoch=epoch,
        )

        if val_metric > best_val_metric:
            print(
                f"Find better metric: val_metric {best_val_metric:.5} -> {val_metric:.5}"
            )
            best_val_metric = val_metric
            checkpoint = create_checkpoint(
                model,
                optimizer,
                epoch,
                scheduler=scheduler,
            )
            torch.save(
                checkpoint,
                f"{cfg.output_dir}/fold{cfg.fold}/best_weights.pth",
            )

    if (epoch + 1) == cfg.epochs:
        # save final best weights, with its distinct name in order to avoid mistakes.
        shutil.copyfile(
            f"{cfg.output_dir}/fold{cfg.fold}/best_weights.pth",
            f"{cfg.output_dir}/fold{cfg.fold}/best_weights_{best_val_metric:.4f}.pth",
        )
