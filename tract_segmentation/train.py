import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cupy as cp
import cv2
import matplotlib.pyplot as plt
import monai
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import seaborn as sns
import torch
from joblib import Parallel, delayed
from monai.data import CSVDataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from tqdm import tqdm

from tract_segmentation.config import CFG
from tract_segmentation.dataset import LitDataModule
from tract_segmentation.module import LitModule
from tract_segmentation.utils import add_3d_paths, create_3d_npy_data


def train(
    random_seed: int = CFG.RANDOM_SEED,
    train_csv_path: str = "train_preprocessed_3d.csv",
    val_fold: str = CFG.VAL_FOLD,
    batch_size: int = CFG.BATCH_SIZE,
    num_workers: int = CFG.NUM_WORKERS,
    learning_rate: float = CFG.LEARNING_RATE,
    weight_decay: float = CFG.WEIGHT_DECAY,
    scheduler: Optional[str] = CFG.SCHEDULER,
    min_lr: float = CFG.MIN_LR,
    gpus: int = CFG.GPUS,
    fast_dev_run: bool = CFG.FAST_DEV_RUN,
    max_epochs: int = CFG.MAX_EPOCHS,
    precision: int = CFG.PRECISION,
    debug: bool = CFG.DEBUG,
):
    pl.seed_everything(random_seed)

    if debug:
        max_epochs = 2

    data_module = LitDataModule(
        train_csv_path=train_csv_path,
        test_csv_path=None,
        val_fold=val_fold,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    module = LitModule(
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        scheduler=scheduler,
        T_max=int(30_000 / batch_size * max_epochs) + 50,
        T_0=25,
        min_lr=min_lr,
    )

    trainer = pl.Trainer(
        fast_dev_run=fast_dev_run,
        gpus=gpus,
        log_every_n_steps=10,
        logger=pl.loggers.CSVLogger(save_dir=CFG.LOGS_PATH),
        max_epochs=max_epochs,
        precision=precision,
    )

    trainer.fit(module, datamodule=data_module)

    if not fast_dev_run:
        trainer.test(module, datamodule=data_module)
        
    return trainer


if os.path.exists("train_preprocessed_3d.csv"):
    train_df = pd.read_csv("train_preprocessed_3d.csv")
else:
    train_df = pd.read_csv(f"{CFG.INPUT_DATA_NPY_DIR}/train_preprocessed.csv")

    if CFG.DEBUG:
        train_df = train_df.head(1_000)

    train_df = add_3d_paths(train_df, stage="train")

    train_df = create_3d_npy_data(train_df, stage="train")

    if CFG.DEBUG:
        print(len(train_df))

    train_df.to_csv("train_preprocessed_3d.csv")

data_module = LitDataModule(
    train_csv_path="train_preprocessed_3d.csv",
    test_csv_path=None,
    val_fold=CFG.VAL_FOLD,
    batch_size=4,
    num_workers=CFG.NUM_WORKERS,
)
data_module.setup()

train_dataloader = data_module.train_dataloader()
batch = next(iter(train_dataloader))





trainer = train()

# From https://www.kaggle.com/code/jirkaborovec?scriptVersionId=93358967&cellId=22
metrics = pd.read_csv(f"{trainer.logger.log_dir}/metrics.csv")[["epoch", "train_loss_epoch", "val_loss"]]
metrics.set_index("epoch", inplace=True)

sns.relplot(data=metrics, kind="line", height=5, aspect=1.5)
plt.grid()
