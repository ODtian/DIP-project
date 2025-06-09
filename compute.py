# compute_metrics.py
import argparse

# import cv2
import os

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image


def dice_score(pred_bin: jnp.ndarray, gt_bin: jnp.ndarray) -> jnp.ndarray | float:
    """Dice = 2|A∩B| / (|A|+|B|)"""
    inter = jnp.sum(pred_bin & gt_bin)
    union = jnp.sum(pred_bin) + jnp.sum(gt_bin)
    return 2 * inter / union if union else 0.0


def rmse(pred: jnp.ndarray, gt: jnp.ndarray) -> jnp.ndarray:
    """Root-Mean-Squared Error"""
    return jnp.sqrt(jnp.mean((pred.astype(jnp.float32) - gt.astype(jnp.float32)) ** 2))


def binarize(img: jnp.ndarray, thresh: int = 1) -> jnp.ndarray:
    """灰度化 + 简单阈值二值化"""
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = jnp.array(Image.fromarray(np.array(img)).convert("L"))  # 转为灰度图
    return (gray > thresh).astype(jnp.uint8)


def parse_args():
    ap = argparse.ArgumentParser(description="Compute Dice & RMSE between two .tif images")
    ap.add_argument("--gt", required=True, help="ground-truth .tif path")
    ap.add_argument("--pred", required=True, help="predicted  .tif path")
    return ap.parse_args()


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)  # Enable 64-bit precision
    args = parse_args()
    assert os.path.exists(args.gt), f"GT file not found: {args.gt}"
    assert os.path.exists(args.pred), f"Pred file not found: {args.pred}"

    gt = jnp.array(Image.open(args.gt))
    pred = jnp.array(Image.open(args.pred))

    # Dice on二值图
    dice = dice_score(binarize(pred), binarize(gt))
    # RMSE on原始像素
    err = rmse(pred, gt)

    print(f"Dice : {dice:.4f}")
    print(f"RMSE : {err:.4f}")
