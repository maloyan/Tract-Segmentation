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

class DiceMetric(Metric):
    def __init__(self):
        super().__init__()

        self.post_processing = monai.transforms.Compose(
            [
                monai.transforms.Activations(sigmoid=True),
                monai.transforms.AsDiscrete(threshold=0.5),
            ]
        )
        self.add_state("dice", default=[])

    def update(self, y_pred, y_true):
        y_pred = self.post_processing(y_pred)
        self.dice.append(monai.metrics.compute_meandice(y_pred, y_true))

    def compute(self):
        return torch.mean(torch.stack(self.dice))