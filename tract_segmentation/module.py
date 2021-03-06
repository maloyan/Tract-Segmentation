import os
from collections import OrderedDict
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
import segmentation_models_pytorch as smp
import torch
from joblib import Parallel, delayed
from monai.data import CSVDataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics import Metric, MetricCollection
from tqdm import tqdm

from tract_segmentation.config import CFG
from tract_segmentation.metrics import DiceMetric


class LitModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float,
        weight_decay: float,
        scheduler: Optional[str],
        T_max: int,
        T_0: int,
        min_lr: int,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.model = self._init_model()

        self.loss_fn = self._init_loss_fn()

        self.metrics = self._init_metrics()

    def _init_model(self):
        model = monai.networks.nets.SwinUNETR(
            in_channels=1,
            out_channels=3,
            img_size=CFG.SPATIAL_SIZE,
            feature_size=48,
            use_checkpoint=True,
        )

        weight = torch.load(
            os.path.join(CFG.PRETRAINED_PATH)
        )

        model.load_from(weights=weight)
        return model
        # return monai.networks.nets.UNet(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=3,
        #     channels=(16, 32, 64, 128, 256),
        #     strides=(2, 2, 2, 2),
        #     num_res_units=2,
        # )

    def _init_loss_fn(self):
        loss_fns = [smp.losses.SoftBCEWithLogitsLoss(), smp.losses.DiceLoss(mode="multilabel")]

        def criterion(y_pred, y_true):
            return sum(loss_fn(y_pred, y_true) for loss_fn in loss_fns) / len(loss_fns)

        return criterion
        #return monai.losses.DiceLoss(sigmoid=True)

    def _init_metrics(self):
        val_metrics = MetricCollection({"val_dice": DiceMetric()})
        test_metrics = MetricCollection({"test_dice": DiceMetric()})

        return torch.nn.ModuleDict(
            {
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
            }
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            params=self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.scheduler is not None:
            if self.hparams.scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.hparams.T_max, eta_min=self.hparams.min_lr
                )
            else:
                raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        else:
            return {"optimizer": optimizer}

    def forward(self, images):
        return self.model(images)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self.shared_step(batch, "test")

    def shared_step(self, batch, stage, log=True):
        images, masks = batch["image_3d"], batch["mask_3d"]
        y_pred = self.model(images)

        loss = self.loss_fn(y_pred, masks)
        if stage != "train":
            metrics = self.metrics[f"{stage}_metrics"](y_pred, masks)
        else:
            metrics = None

        if log:
            batch_size = images.shape[0]
            self._log(loss, batch_size, metrics, stage)

        return loss

    def _log(self, loss, batch_size, metrics, stage):
        on_step = True if stage == "train" else False

        self.log(
            f"{stage}_loss",
            loss,
            on_step=on_step,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        if metrics is not None:
            self.log_dict(
                metrics, on_step=on_step, on_epoch=True, batch_size=batch_size
            )

    @classmethod
    def load_eval_checkpoint(cls, checkpoint_path, device):
        module = cls.load_from_checkpoint(checkpoint_path=checkpoint_path).to(device)
        module.eval()

        return module
