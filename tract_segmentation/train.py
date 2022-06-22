import copy
from glob import glob
import time
from collections import defaultdict

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.optim as optim
import wandb
from sklearn.model_selection import StratifiedGroupKFold
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from tract_segmentation.config import CFG
from tract_segmentation.dataset import BuildDataset
from tract_segmentation.engine import train_one_epoch, valid_one_epoch
from tract_segmentation.utils import path2info, set_seed

set_seed(CFG.seed)
run = wandb.init(
    project="uw-maddison-gi-tract",
    config={k: v for k, v in dict(vars(CFG)).items() if "__" not in k},
    name=f"fold-0|dim-{CFG.img_size[0]}x{CFG.img_size[1]}|model-{CFG.model_name}",
    group=CFG.comment,
)

BASE_PATH = "/kaggle/input/uw-madison-gi-tract-image-segmentation"

# paths = glob(
#     f"/kaggle/input/uw-madison-gi-tract-image-segmentation/train/**/*png",
#     recursive=True,
# )
# path_df = pd.DataFrame(paths, columns=["image_path"])
# path_df = path_df.apply(lambda x: path2info(x), axis=1)

# import IPython; IPython.embed(); exit(1)
df = pd.read_csv(CFG.data_path)
df["segmentation"] = df.segmentation.fillna("")
df["rle_len"] = df.segmentation.map(len)  # length of each rle mask
df["mask_path"] = df.mask_path.str.replace("/png/", "/np").str.replace(".png", ".npy")


df2 = (
    df.groupby(["id"])["segmentation"].agg(list).to_frame().reset_index()
)  # rle list of each id
df2 = df2.merge(
    df.groupby(["id"])["rle_len"].agg(sum).to_frame().reset_index()
)  # total length of all rles of each id

df = df.drop(columns=["segmentation", "class", "rle_len"])
df = df.groupby(["id"]).head(1).reset_index(drop=True)
df = df.merge(df2, on=["id"])
df["empty"] = df.rle_len == 0  # empty masks


skf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
for fold, (train_idx, val_idx) in enumerate(
    skf.split(df, df["empty"], groups=df["case"])
):
    df.loc[val_idx, "fold"] = fold

data_transforms = {
    "train": A.Compose(
        [
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            #         A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.5
            ),
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                    # #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                ],
                p=0.25,
            ),
            A.CoarseDropout(
                max_holes=8,
                max_height=CFG.img_size[0] // 20,
                max_width=CFG.img_size[1] // 20,
                min_holes=5,
                fill_value=0,
                mask_fill_value=0,
                p=0.5,
            ),
        ],
        p=1.0,
    ),
    "valid": A.Compose(
        [
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
        ],
        p=1.0,
    ),
}


train_df = df.query("fold!=0").reset_index(drop=True)

train_non_empty = train_df[train_df["empty"] == False].reset_index(drop=True)
train_empty = (
    train_df[train_df["empty"] == True]
    .sample(int(train_non_empty.shape[0] * 0.66))
    .reset_index(drop=True)
)
train_df = pd.concat([train_non_empty, train_empty], axis=0)

valid_df = df.query("fold==0").reset_index(drop=True)

train_dataset = BuildDataset(train_df, transforms=data_transforms["train"])
valid_dataset = BuildDataset(valid_df, transforms=data_transforms["valid"])

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.train_bs,
    num_workers=4,
    shuffle=True,
    pin_memory=True,
    drop_last=False,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=CFG.valid_bs,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
)

model = smp.UnetPlusPlus(
    encoder_name=CFG.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=CFG.num_classes,  # model output channels (number of classes in your dataset)
    activation=None,
)

model = torch.nn.DataParallel(model, device_ids=CFG.device_ids)

model.to(CFG.device)

optimizer = optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
# scheduler = lr_scheduler.ReduceLROnPlateau(
#     optimizer,
#     mode="min",
#     factor=0.1,
#     patience=7,
#     threshold=0.0001,
#     min_lr=CFG.min_lr,
# )

scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=CFG.T_0, eta_min=CFG.min_lr
)


# To automatically log gradients
wandb.watch(model, log_freq=100)

if torch.cuda.is_available():
    print("cuda: {}\n".format(torch.cuda.get_device_name()))

start = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_dice = -np.inf
best_epoch = -1
history = defaultdict(list)

for epoch in range(1, CFG.epochs + 1):
    print(f"Epoch {epoch}/{CFG.epochs}", end="")
    train_loss = train_one_epoch(
        model,
        optimizer,
        scheduler,
        dataloader=train_loader,
        device=CFG.device,
        epoch=epoch,
    )

    val_loss, val_scores, bg_img, true_mask, pred_mask = valid_one_epoch(
        model, valid_loader, device=CFG.device, epoch=epoch
    )
    val_dice, val_jaccard = val_scores

    history["Train Loss"].append(train_loss)
    history["Valid Loss"].append(val_loss)
    history["Valid Dice"].append(val_dice)
    history["Valid Jaccard"].append(val_jaccard)

    rand_img = np.random.randint(pred_mask.shape[0])
    res = torch.zeros(pred_mask[rand_img].shape[1:])
    true = torch.zeros(true_mask[rand_img].shape[1:])

    for ind, i in enumerate(pred_mask[rand_img].detach().cpu().numpy()):
        res += (ind + 1) * (i > CFG.threshold)

    for ind, i in enumerate(true_mask[rand_img].detach().cpu().numpy()):
        true += (ind + 1) * (i > CFG.threshold)
    wandb.log(
        {
            "Train Loss": train_loss,
            "Valid Loss": val_loss,
            "Valid Dice": val_dice,
            "Valid Jaccard": val_jaccard,
            "LR": optimizer.param_groups[0]["lr"],
            "Image": wandb.Image(
                bg_img[rand_img],
                masks={
                    "prediction": {
                        "mask_data": res.numpy(),
                        "class_labels": CFG.labels,
                    },
                    "ground truth": {
                        "mask_data": true.numpy(),
                        "class_labels": CFG.labels,
                    },
                },
            ),
        }
    )

    print(f"Valid Dice: {val_dice:0.4f} | Valid Jaccard: {val_jaccard:0.4f}")

    # deep copy the model
    if val_dice >= best_dice:
        print(f"Valid Score Improved ({best_dice:0.4f} ---> {val_dice:0.4f})")
        best_dice = val_dice
        best_jaccard = val_jaccard
        best_epoch = epoch
        run.summary["Best Dice"] = best_dice
        run.summary["Best Jaccard"] = best_jaccard
        run.summary["Best Epoch"] = best_epoch
        best_model_wts = copy.deepcopy(model.state_dict())
        torch.save(model.state_dict(), f"{CFG.checkpoints}_best.pt")
        # Save a model file from the current directory

    last_model_wts = copy.deepcopy(model.state_dict())
    torch.save(model.state_dict(), f"{CFG.checkpoints}_last.pt")

    print()
    print()

end = time.time()
time_elapsed = end - start
print(
    "Training complete in {:.0f}h {:.0f}m {:.0f}s".format(
        time_elapsed // 3600,
        (time_elapsed % 3600) // 60,
        (time_elapsed % 3600) % 60,
    )
)
print("Best Score: {:.4f}".format(best_jaccard))
run.finish()
