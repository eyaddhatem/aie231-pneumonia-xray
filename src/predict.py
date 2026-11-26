"""Inference: one image (or a folder) -> P(pneumonia) and a label. Uses the SAME model-internal scaling as training."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.models import IMG_SIZE, build_vgg

CLASSES = ["NORMAL", "PNEUMONIA"]
DEFAULT_THRESHOLD = 0.15  # screening threshold chosen on validation (favours recall on pneumonia)


def load_image(path) -> np.ndarray:
    im = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.asarray(im, dtype=np.float32)[None]  # [0,255], batch of 1


def load_model(weights_path):
    model, _ = build_vgg(0.5, pretrained=False)
    model.load_weights(weights_path)
    return model


def predict(model, path, threshold: float = DEFAULT_THRESHOLD) -> dict:
    p = float(model.predict(load_image(path), verbose=0).ravel()[0])
    return dict(file=str(path), p_pneumonia=round(p, 4), label=CLASSES[int(p >= threshold)], threshold=threshold)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("image_or_folder")
    ap.add_argument("--weights", default="models/vgg16_final.weights.h5")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    a = ap.parse_args()
    m = load_model(a.weights)
    src = Path(a.image_or_folder)
    files = sorted(src.glob("*.jp*g")) + sorted(src.glob("*.png")) if src.is_dir() else [src]
    for f in files:
        print(predict(m, f, a.threshold))
