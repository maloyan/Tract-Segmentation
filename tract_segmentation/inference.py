from glob import glob

import albumentations as A
import cupy as cp
import cv2
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class CFG:
    seed = 101
    debug = False  # set debug=False for Full Training
    exp_name = "Baselinev2"
    model_name = "Unet++"
    backbone = "efficientnet-b6"
    img_size = [384, 384]
    comment = f"{model_name}-{backbone}-{img_size[0]}x{img_size[1]}"
    train_bs = 32
    valid_bs = train_bs * 2
    epochs = 15
    lr = 2e-3
    scheduler = "CosineAnnealingLR"
    min_lr = 1e-6
    T_max = int(30000 / train_bs * epochs) + 50
    T_0 = 25
    warmup_epochs = 0
    wd = 1e-6
    n_accumulate = max(1, 32 // train_bs)
    n_fold = 5
    num_classes = 3
    device = "cuda"
    device_ids = [0]
    data_path = "/kaggle/input/uwmgi-mask-dataset/train.csv"
    labels = {1: "large_bowel", 2: "small_bowel", 3: "stomach"}
    threshold = 0.45
    checkpoints = f'/kaggle/input/tract_segmentation/{comment}'


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
    """
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    """
    msk = cp.array(msk)
    pixels = msk.flatten()
    pad = cp.array([0])
    pixels = cp.concatenate([pad, pixels, pad])
    runs = cp.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return " ".join(str(x) for x in runs)


def masks2rles(msks, ids, heights, widths):
    pred_strings = []
    pred_ids = []
    pred_classes = []
    for idx in range(msks.shape[0]):
        height = heights[idx].item()
        width = widths[idx].item()
        msk = cv2.resize(
            msks[idx], dsize=(width, height), interpolation=cv2.INTER_NEAREST
        )  # back to original shape
        rle = [None] * 3
        for midx in [0, 1, 2]:
            rle[midx] = mask2rle(msk[..., midx])
        pred_strings.extend(rle)
        pred_ids.extend([ids[idx]] * len(rle))
        pred_classes.extend(["large_bowel", "small_bowel", "stomach"])
    return pred_strings, pred_ids, pred_classes


class BuildDataset(torch.utils.data.Dataset):
    def __init__(self, df, label=True, transforms=None):
        self.df = df
        self.label = label
        self.img_paths = df["image_path"].tolist()
        self.ids = df["id"].tolist()
        if label:
            self.msk_paths = df["mask_path"].tolist()
        else:
            self.msk_paths = None
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        id_ = self.ids[index]
        img = []
        img = load_img(img_path)
        h, w = img.shape[:2]

        if self.label:
            msk_path = self.msk_paths[index]
            msk = load_msk(msk_path)
            if self.transforms:
                data = self.transforms(image=img, mask=msk)
                img = data["image"]
                msk = data["mask"]
            img = np.transpose(img, (2, 0, 1))
            msk = np.transpose(msk, (2, 0, 1))
            return torch.tensor(img), torch.tensor(msk)
        else:
            if self.transforms:
                data = self.transforms(image=img)
                img = data["image"]
            img = np.transpose(img, (2, 0, 1))
            return torch.tensor(img), id_, h, w


BASE_PATH = "/kaggle/input/uw-madison-gi-tract-image-segmentation"
CKPT_DIR = "/kaggle/input/uwmgi-unet-train-pytorch-ds"

sub_df = pd.read_csv(
    "/kaggle/input/uw-madison-gi-tract-image-segmentation/sample_submission.csv"
)

if not len(sub_df):
    debug = True
    sub_df = pd.read_csv(
        "/kaggle/input/uw-madison-gi-tract-image-segmentation/train.csv"
    )[196:200]
    sub_df = sub_df.drop(columns=["class", "segmentation"]).drop_duplicates()
else:
    debug = False
    sub_df = sub_df.drop(columns=["class", "predicted"]).drop_duplicates()
sub_df = sub_df.apply(get_metadata, axis=1)

if debug:
    paths = glob(
        f"/kaggle/input/uw-madison-gi-tract-image-segmentation/train/**/*png",
        recursive=True,
    )
else:
    paths = glob(
        f"/kaggle/input/uw-madison-gi-tract-image-segmentation/test/**/*png",
        recursive=True,
    )


path_df = pd.DataFrame(paths, columns=["image_path"])
path_df = path_df.apply(path2info, axis=1)

test_df = sub_df.merge(path_df, on=["case", "day", "slice"], how="left")

data_transforms = {
    "train": A.Compose(
        [
            A.Resize(*CFG.img_size, interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            #         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=5, p=0.5),
            A.OneOf(
                [
                    A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                    # #             A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=1.0),
                    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                ],
                p=0.25,
            ),
            #         A.CoarseDropout(max_holes=8, max_height=CFG.img_size[0]//20, max_width=CFG.img_size[1]//20,
            #                          min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
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

test_dataset = BuildDataset(test_df, label=False, transforms=data_transforms["valid"])
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.valid_bs,
    num_workers=4,
    shuffle=False,
    pin_memory=False,
)
model = smp.UnetPlusPlus(
    encoder_name=CFG.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights=None,  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=CFG.num_classes,  # model output channels (number of classes in your dataset)
    activation=None,
)

model = torch.nn.DataParallel(model, device_ids=CFG.device_ids)

model.to(CFG.device)

model.load_state_dict(torch.load(f"{CFG.checkpoints}_best.pt"))
model.eval()

with torch.no_grad():
    msks = []
    imgs = []
    pred_strings = []
    pred_ids = []
    pred_classes = []
    for img, ids, heights, widths in tqdm(test_loader, total=len(test_loader), desc="Infer "):
        img = img.to(CFG.device, dtype=torch.float)  # .squeeze(0)
        msk = model(img)
        msk = nn.Sigmoid()(msk)
        msk = (
            (msk.permute((0, 2, 3, 1)) > CFG.threshold)
            .to(torch.uint8)
            .cpu()
            .detach()
            .numpy()
        )
        result = masks2rles(msk, ids, heights, widths)
        pred_strings.extend(result[0])
        pred_ids.extend(result[1])
        pred_classes.extend(result[2])

pred_df = pd.DataFrame({
    "id":pred_ids,
    "class":pred_classes,
    "predicted":pred_strings
})
if not debug:
    sub_df = pd.read_csv('/kaggle/input/uw-madison-gi-tract-image-segmentation/sample_submission.csv')
    del sub_df['predicted']
else:
    sub_df = pd.read_csv('/kaggle/input/uw-madison-gi-tract-image-segmentation/train.csv')[:1000*3]
    del sub_df['segmentation']
    
sub_df = sub_df.merge(pred_df, on=['id','class'])
sub_df.to_csv('submission.csv',index=False)
