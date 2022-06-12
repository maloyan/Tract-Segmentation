from pathlib import Path

import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm


def add_3d_paths(df, stage):
    df["image_3d"] = df["image_path"].str.split("/scans").str[0] + "_image_3d.npy"
    df["image_3d"] = df["image_3d"].str.replace("input", "working")

    if stage == "train":
        df["mask_3d"] = df["image_3d"].str.replace("_image_", "_mask_")

    return df


def load_image(path):
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)  # uint16
    return image


def load_mask(row):
    shape = (row.height, row.width, 3)
    mask = np.zeros(shape, dtype=np.uint8)

    rles = eval(row.segmentation.replace("nan", "''"))
    for i, rle in enumerate(rles):
        if rle:
            mask[..., i] = rle_decode(rle, shape[:2])

    return mask * 255


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background

    """
    s = np.asarray(mask_rle.split(), dtype=int)
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    mask = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1

    return mask.reshape(shape)  # Needed to align to RLE direction


def create_3d_image_mask(group_df, stage):
    image_3d, mask_3d = [], []
    for row in group_df.itertuples():
        image_3d.append(load_image(row.image_path))  # uint16

        if stage == "train":
            mask_3d.append(load_mask(row))  # uint8

    image_3d = np.stack(image_3d, axis=-1)

    dir_3d = Path(row.image_3d).parent
    dir_3d.mkdir(parents=True, exist_ok=True)
    np.save(row.image_3d, image_3d)

    if stage == "train":
        mask_3d = np.stack(mask_3d, axis=-1)
        np.save(row.mask_3d, mask_3d)

    return group_df.id.to_list()


def create_3d_npy_data(df, stage, num_workers):
    grouped = df.groupby(["case", "day"])
    ids = Parallel(n_jobs=num_workers)(
        delayed(create_3d_image_mask)(group_df, stage)
        for _, group_df in tqdm(
            grouped, total=len(grouped), desc="Iterating over case-day groups"
        )
    )

    columns_to_drop = ["id", "slice", "image_path"]
    if stage == "train":
        columns_to_drop += [
            "classes",
            "segmentation",
            "rle_len",
            "empty",
            "mask_path",
            "image_paths",
        ]

    df = df.drop(columns=columns_to_drop)
    df = df.drop_duplicates().reset_index(drop=True)
    df["ids"] = ids

    return df
