from typing import Callable, Optional, Tuple

import monai
import pandas as pd
import pytorch_lightning as pl
from monai.data import CSVDataset, DataLoader
from torch.utils.data import DataLoader


class LitDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv_path: str,
        test_csv_path: Optional[str],
        val_fold: int,
        batch_size: int,
        num_workers: int,
        spatial_size: Tuple[int],
    ):
        super().__init__()

        self.save_hyperparameters()

        self.train_df = pd.read_csv(train_csv_path)

        if test_csv_path is not None:
            self.test_df = pd.read_csv(test_csv_path)
        else:
            self.test_df = None

        (
            self.train_transforms,
            self.val_transforms,
            self.test_transforms,
        ) = self._init_transforms()

    def _init_transforms(self):
        spatial_size = self.hparams.spatial_size

        transforms = [
            monai.transforms.LoadImaged(keys=["image_3d", "mask_3d"]),
            monai.transforms.AddChanneld(keys=["image_3d"]),
            monai.transforms.AsChannelFirstd(keys=["mask_3d"], channel_dim=2),
            monai.transforms.ScaleIntensityd(keys=["image_3d", "mask_3d"]),
            # monai.transforms.ResizeWithPadOrCrop(keys=["image_3d", "mask_3d"], spatial_size=spatial_size),
            monai.transforms.Resized(
                keys=["image_3d", "mask_3d"], spatial_size=spatial_size, mode="nearest"
            ),
        ]

        test_transforms = [
            monai.transforms.LoadImaged(keys=["image_3d"]),
            monai.transforms.AddChanneld(keys=["image_3d"]),
            monai.transforms.ScaleIntensityd(keys=["image_3d"]),
            # monai.transforms.ResizeWithPadOrCrop(keys=["image_3d"], spatial_size=spatial_size),
            monai.transforms.Resized(
                keys=["image_3d"], spatial_size=spatial_size, mode="nearest"
            ),
        ]

        train_transforms = monai.transforms.Compose(transforms)
        val_transforms = monai.transforms.Compose(transforms)
        test_transforms = monai.transforms.Compose(test_transforms)

        return train_transforms, val_transforms, test_transforms

    def setup(self, stage: Optional[str] = None):
        train_df = self.train_df[
            self.train_df.fold != self.hparams.val_fold
        ].reset_index(drop=True)
        val_df = self.train_df[self.train_df.fold == self.hparams.val_fold].reset_index(
            drop=True
        )

        if stage == "fit" or stage is None:
            self.train_dataset = self._dataset(
                train_df, transforms=self.train_transforms
            )
            self.val_dataset = self._dataset(val_df, transforms=self.val_transforms)

        if stage == "test" or stage is None:
            if self.test_df is not None:
                self.test_dataset = self._dataset(
                    self.test_df, transforms=self.test_transforms
                )
            else:
                self.test_dataset = self._dataset(
                    val_df, transforms=self.val_transforms
                )

    def _dataset(self, df: pd.DataFrame, transforms: Callable) -> CSVDataset:
        return CSVDataset(src=df, transform=transforms)

    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset)

    def _dataloader(self, dataset: CSVDataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=train,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
        )
