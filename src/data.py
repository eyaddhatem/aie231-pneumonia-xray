"""Data loading and inventory for the Chest X-Ray (Pneumonia) project.

Milestone 1 only needs inventory + EDA helpers. Splitting and the tf.data
pipeline are added in Milestone 2 in this same module.
"""
import hashlib
import os
from pathlib import Path

import pandas as pd
from PIL import Image

DATASET = "paultimothymooney/chest-xray-pneumonia"
CLASSES = ["NORMAL", "PNEUMONIA"]  # label 0 = NORMAL, label 1 = PNEUMONIA (positive class)
IMG_EXT = {".jpeg", ".jpg", ".png"}


def download(dataset: str = DATASET) -> str:
    """Download the dataset with kagglehub and return the local folder."""
    import kagglehub
    return kagglehub.dataset_download(dataset)


def find_data_root(base: str) -> Path:
    """Find the folder that directly contains train/, test/ and val/.

    The Kaggle zip nests the data (chest_xray/chest_xray/...) and also holds a
    __MACOSX copy, so we search instead of hard-coding a path.
    """
    for p in sorted(Path(base).rglob("train")):
        if p.is_dir() and "__MACOSX" not in p.parts and (p.parent / "test").is_dir():
            return p.parent
    raise FileNotFoundError(f"No train/test folders under {base}")


def file_md5(path) -> str:
    """MD5 of the raw file bytes, used to detect exact duplicate images."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_inventory(root, with_hash: bool = True) -> pd.DataFrame:
    """One row per image: path, original split, label, size, mode, file size, hash.

    The label comes from the parent folder name. Pneumonia files are named
    person<ID>_bacteria_<n> / person<ID>_virus_<n>, which gives a subtype and a
    patient id (Normal files are IM-xxxx-yyyy / NORMAL2-IM-xxxx-yyyy and carry no
    patient id we can rely on).
    """
    rows = []
    for split in ["train", "val", "test"]:
        for cls in CLASSES:
            d = Path(root) / split / cls
            if not d.is_dir():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() not in IMG_EXT or p.name.startswith("."):
                    continue
                with Image.open(p) as im:
                    w, h = im.size
                    mode = im.mode
                name = p.name.lower()
                subtype = "bacteria" if "bacteria" in name else "virus" if "virus" in name else "normal"
                patient = name.split("_")[0] if name.startswith("person") else None
                rows.append(dict(
                    path=str(p), orig_split=split, label_name=cls, label=CLASSES.index(cls),
                    subtype=subtype, patient_id=patient, width=w, height=h, mode=mode,
                    kb=os.path.getsize(p) / 1024,
                    md5=file_md5(p) if with_hash else None))
    df = pd.DataFrame(rows)
    df["aspect"] = df["width"] / df["height"]
    df["megapixels"] = df["width"] * df["height"] / 1e6
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--root", default=None)
    a = ap.parse_args()
    base = download() if a.download else a.root
    root = find_data_root(base)
    inv = build_inventory(root)
    print(root, len(inv))
    print(inv.groupby(["orig_split", "label_name"]).size())


# ---------------------------------------------------------------------------
# Milestone 2: cleaning, leak-free splitting, class weights, tf.data pipeline
# ---------------------------------------------------------------------------
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

IMG_SIZE = 224
SEED = 42


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one copy of every byte-identical image (same MD5).

    If the same bytes carry different labels the label is ambiguous, so all copies are dropped.
    """
    lab_per_hash = df.groupby("md5")["label"].nunique()
    conflicting = lab_per_hash[lab_per_hash > 1].index
    out = df[~df["md5"].isin(conflicting)]
    return out.drop_duplicates("md5", keep="first").reset_index(drop=True)


def add_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add `group`: images of the same patient must share a group.

    Pneumonia files carry a patient id in the name. NORMAL files do not, so each
    NORMAL image is its own group (limitation: a NORMAL patient may still span splits).
    """
    df = df.copy()
    df["group"] = np.where(df["patient_id"].notna(), "P_" + df["patient_id"].astype(str), "I_" + df["md5"])
    return df


def make_splits(df: pd.DataFrame, seed: int = SEED, n_folds: int = 20,
                n_test: int = 3, n_val: int = 3) -> pd.DataFrame:
    """Stratified, group-aware 70/15/15 split (14 train folds, 3 val, 3 test of 20).

    StratifiedGroupKFold keeps the class ratio in every fold AND never puts one group
    (patient) in two folds. Cutting 20 folds then merging them gives ~70/15/15.
    """
    df = add_groups(df)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1)
    for k, (_, idx) in enumerate(sgkf.split(df, df["label"], df["group"])):
        fold[idx] = k
    df["split"] = np.where(fold < n_test, "test", np.where(fold < n_test + n_val, "val", "train"))
    return df


def check_no_leakage(df: pd.DataFrame) -> None:
    """Raise AssertionError if any group, file or hash appears in two splits."""
    for col in ["group", "md5", "path"]:
        n = df.groupby(col)["split"].nunique()
        assert (n <= 1).all(), f"Leakage: {int((n > 1).sum())} {col} values span several splits"
    assert set(df["split"]) == {"train", "val", "test"}


def class_weights(train_df: pd.DataFrame) -> dict:
    """Balanced weights w_c = N / (n_classes * n_c), computed from TRAIN only."""
    counts = train_df["label"].value_counts().sort_index()
    n, k = len(train_df), len(counts)
    return {int(c): float(n / (k * m)) for c, m in counts.items()}


def make_dataset(df: pd.DataFrame, batch_size: int = 32, img_size: int = IMG_SIZE,
                 training: bool = False, augment: bool = False, seed: int = SEED):
    """tf.data pipeline: file -> 3-channel float32 [0,255] resized image, label.

    Pixels are left in [0,255]; the *model* rescales them (Rescaling(1/255) for the custom CNN,
    preprocess_input for VGG16/ResNet-50). That keeps training and inference preprocessing identical.
    Augmentation (flip is deliberately excluded for X-rays) runs on the TRAIN set only.
    """
    import tensorflow as tf

    def load(path, y):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (img_size, img_size))
        return tf.cast(img, tf.float32), tf.cast(y, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["label"].values))
    if training:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if augment:
        aug = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.04, seed=seed),
            tf.keras.layers.RandomZoom(0.1, seed=seed),
            tf.keras.layers.RandomContrast(0.15, seed=seed),
        ])
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def load_flat_arrays(df: pd.DataFrame, size: int = 64) -> np.ndarray:
    """Load images as flat grey vectors (size*size) in [0,1] for the classical baselines."""
    out = np.empty((len(df), size * size), dtype=np.float32)
    for i, p in enumerate(df["path"].values):
        with Image.open(p) as im:
            out[i] = np.asarray(im.convert("L").resize((size, size)), dtype=np.float32).ravel() / 255.0
    return out


def rebase_paths(df: pd.DataFrame, root) -> pd.DataFrame:
    """Point `path` at the current dataset folder.

    Colab starts a fresh machine every session, so absolute paths saved in a CSV from an
    earlier session are stale. The layout <split>/<class>/<file> is stable, so rebuild from it.
    """
    df = df.copy()
    split_col = "orig_split"
    df["path"] = [str(Path(root) / s / c / Path(p).name)
                  for s, c, p in zip(df[split_col], df["label_name"], df["path"])]
    return df
