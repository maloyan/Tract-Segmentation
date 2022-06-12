import os.path

import pandas as pd
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pkg_resources import resource_filename

from tract_segmentation.dataset import LitDataModule
from tract_segmentation.model import LitModule
from tract_segmentation.utils import add_3d_paths, create_3d_npy_data

def main():
    config = OmegaConf.load(resource_filename(__name__, "configs/config.yml"))

    if os.path.exists("train_preprocessed_3d.csv"):
        train_df = pd.read_csv(f"train_preprocessed_3d.csv")
    else:
        train_df = pd.read_csv(config.intpu_data_npy_dir / "train_preprocessed.csv")
        train_df = add_3d_paths(train_df, stage="train")
        train_df = create_3d_npy_data(
            train_df, stage="train", num_workers=config.num_workers
        )
        train_df.to_csv(f"train_preprocessed_3d.csv")

    pl.seed_everything(config.random_seed)


    data_module = LitDataModule(
        train_csv_path="train_preprocessed_3d.csv",
        test_csv_path=None,
        val_fold=config.val_fold,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        spatial_size=config.spatial_size,
    )

    module = LitModule(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        scheduler=config.scheduler,
        T_max=int(30_000 / config.batch_size * config.max_epochs) + 50,
        T_0=25,
        min_lr=config.min_lr,
    )

    trainer = pl.Trainer(
        accelerator=config.accelerator,
        devices=config.gpus,
        log_every_n_steps=10,
        logger=pl.loggers.WandbLogger(project="tract_segmentation"),
        max_epochs=config.max_epochs,
        precision=config.precision,
        fast_dev_run=config.fast_dev_run,
    )

    trainer.fit(module, datamodule=data_module)
if __name__ == '__main__':
    main()