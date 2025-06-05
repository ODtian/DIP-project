# -*- coding: utf-8 -*-
"""
interp_2x2d_rbf.py  – 2×2 D RBF interpolation for CS270 Project-4   (2025-06-05)
-------------------------------------------------------------------------------
• 支持 “低帧率” (--down_fps) 与 “短采集时长” (--cr) 两种模式；
• 默认 Multiquadric 核 φ(r)=√(1+(εr)²)，ε≈1e4–3e4；
• 立方体维度约定： (Z, X, T)  ↔  (深度, 横向, 帧) —— 与 PALA 原始 IQ 一致。

python interp_2x2d_rbf.py `
  --mats_path "datasets/PALA_data_InVivoRatBrain_part1/PALA_data_InVivoRatBrain/IQ" `
  --out_path  "datasets/PALA_data_InVivoRatBrain_part1/PALA_data_InVivoRatBrain/Results/DS10_100Hz" `
  --orig_fps 1000 `
  --down_fps 100 `
  --gpu
"""
import argparse
import os
from typing import Tuple

import numpy as np
import scipy.io as sio
import torch
from tqdm import tqdm


# ---------------- RBF helpers -------------------------------------------------
def mq_kernel(dist: torch.Tensor, eps: float) -> torch.Tensor:
    return torch.sqrt(1.0 + (eps * dist) ** 2)


def solve_coeff(coords: torch.Tensor, values: torch.Tensor, eps: float, lam: float) -> torch.Tensor:
    r = torch.cdist(coords, coords)
    phi = mq_kernel(r, eps)
    if lam > 0:
        phi += lam * torch.eye(phi.size(0), device=coords.device)
    return torch.linalg.solve(phi, values)


# ---------------- 2-D slice interpolation ------------------------------------
def rbf2d_fill(grid: np.ndarray, eps: float, lam: float, device: str, batch: int) -> np.ndarray:
    if not np.isnan(grid).any():  # 已满帧则直接返回
        return grid.copy()

    known_idx = np.argwhere(~np.isnan(grid))
    if known_idx.size == 0:  # 整片空，直接返回 NaN
        return grid.copy()

    vals = grid[~np.isnan(grid)].astype(np.float32)
    h, w = grid.shape
    norm = known_idx.astype(np.float32)
    norm[:, 0] /= h
    norm[:, 1] /= w  # 归一化坐标

    c_t = torch.tensor(norm, device=device)
    v_t = torch.tensor(vals, device=device)
    w_t = solve_coeff(c_t, v_t, eps, lam)  # 权重

    # 目标网格
    ij_all = np.indices((h, w)).reshape(2, -1).T
    norm_all = ij_all.astype(np.float32)
    norm_all[:, 0] /= h
    norm_all[:, 1] /= w
    p_t = torch.tensor(norm_all, device=device)

    flat_out = grid.ravel()
    for s in range(0, p_t.size(0), batch):
        e = min(s + batch, p_t.size(0))
        phi_blk = mq_kernel(torch.cdist(p_t[s:e], c_t), eps)
        est = (phi_blk @ w_t).cpu().numpy()
        flat_out[np.ravel_multi_index((ij_all[s:e, 0], ij_all[s:e, 1]), grid.shape)] = est

    return flat_out.reshape(grid.shape)


# ---------------- 2×2 D cube interpolation -----------------------------------
def interp_cube(iq_nan: np.ndarray, eps: float, lam: float, device: str, batch: int) -> np.ndarray:
    z, x, t = iq_nan.shape  # (Z, X, T)
    real, imag = iq_nan.real.copy(), iq_nan.imag.copy()

    def _fill(channel: np.ndarray) -> np.ndarray:
        rec_xt = np.empty_like(channel)
        for zi in tqdm(range(z), desc="x-t", leave=False):
            rec_xt[zi] = rbf2d_fill(channel[zi], eps, lam, device, batch)  # (X,T)

        rec_zt = np.empty_like(channel)
        for xi in tqdm(range(x), desc="z-t", leave=False):
            rec_zt[:, xi, :] = rbf2d_fill(channel[:, xi, :].T, eps, lam, device, batch).T
        return 0.5 * (rec_xt + rec_zt)

    return _fill(real) + 1j * _fill(imag)


# ---------------- CLI ---------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mats_path", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--orig_fps", type=int, default=1000)
    # 二选一： --down_fps 或 --cr
    ap.add_argument("--down_fps", type=int, help="target frame-rate (low-FPS mode)")
    ap.add_argument("--cr", type=int, help="compression ratio (short acquisition mode)")
    ap.add_argument("--eps", type=float, default=1e4)
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=16384)
    ap.add_argument("--gpu", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    device = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
    assert (args.down_fps is None) ^ (args.cr is None), "必须二选一：--down_fps 或 --cr"

    if args.down_fps:  # 低帧率模式
        ds = args.orig_fps // args.down_fps
        assert ds >= 2 and args.orig_fps % args.down_fps == 0, "--down_fps 必须整除 orig_fps"
    os.makedirs(args.out_path, exist_ok=True)

    for fname in sorted(p for p in os.listdir(args.mats_path) if p.lower().endswith(".mat")):
        in_path = os.path.join(args.mats_path, fname)
        print(f"[→] {fname}")
        mat = sio.loadmat(in_path)
        iq_full = mat["IQ"].astype(np.complex64)
        uf, pdata = mat.get("UF"), mat.get("PData")

        # cube = np.full_like(iq_full, np.nan, dtype=np.complex64)
        # if args.down_fps:
        #     cube[:, :, ::ds] = iq_full[:, :, ::ds]
        # else:                                     # CR 模式
        #     first_N = iq_full.shape[2] // args.cr
        #     cube[:, :, :first_N] = iq_full[:, :, :first_N]
        cube = np.full_like(iq_full, np.nan, dtype=np.complex64)
        if args.down_fps:
            ds = args.orig_fps // args.down_fps
            cube[:, :, ::ds] = iq_full[:, :, ::ds]
        else:  # CR 模式
            first_N = iq_full.shape[2] // args.cr
            cube[:, :, :first_N] = iq_full[:, :, :first_N]

        iq_rec = interp_cube(cube, args.eps, args.lam, device, args.batch)
        out_f = os.path.join(args.out_path, fname)
        sio.savemat(out_f, {"IQ": iq_rec, "UF": uf, "PData": pdata})
        print(f"[✓] saved → {out_f}")


if __name__ == "__main__":
    main()
