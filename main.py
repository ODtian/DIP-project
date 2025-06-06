from functools import partial
from os import PathLike
from pathlib import Path
import time
from typing import Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio


# --- 复用之前的 multiquadric_kernel 函数 ---
def multiquadric_kernel(r, epsilon):
    return jnp.sqrt(1 + (epsilon * r) ** 2)


# --- 核心JIT函数：对单个2D切片进行插值 ---
# 这个函数现在接受预先计算好的矩阵和索引，使其与JIT兼容


# @jax.jit
def rbf_2d(inv_phi: jnp.ndarray, new_phi: jnp.ndarray, know_values: jnp.ndarray):
    weights = inv_phi @ know_values.reshape(-1)
    # weights = jnp.linalg.solve(inv_phi, know_values.reshape(-1))
    # print("weight", weights)
    # .flatten()
    reconstructed_values = new_phi @ weights
    return reconstructed_values


@jax.jit
def solve_phi(
    known_coords: jnp.ndarray, unknown_coords: jnp.ndarray, epsilon: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dist_known = jnp.linalg.norm(known_coords[:, None, :] - known_coords[None, :, :], axis=-1)
    phi = multiquadric_kernel(dist_known, epsilon) + 1e-4 * jnp.eye(known_coords.shape[0])
    inv_phi = jnp.linalg.inv(phi)  # 预计算逆矩阵

    new_dist = jnp.linalg.norm(unknown_coords[:, None, :] - known_coords[None, :, :], axis=-1)
    new_phi = multiquadric_kernel(new_dist, epsilon)
    return inv_phi, new_phi


@jax.jit
def solve(known_coords: jnp.ndarray, unknown_coords: jnp.ndarray, know_values: jnp.ndarray, epsilon: float):

    # Calculate squared Euclidean distances for known_coords vs known_coords
    # known_coords shape: (N, D)
    sum_known_sq = jnp.sum(known_coords**2, axis=1, keepdims=True)  # Shape: (N, 1)
    dot_known = jnp.matmul(known_coords, known_coords.T)  # Shape: (N, N)
    # dist_known_sq = sum_A_sq - 2*A@B.T + sum_B_sq.T
    dist_known_sq = sum_known_sq - 2 * dot_known + sum_known_sq.T  # Shape: (N, N)
    # Add a small epsilon or ensure non-negativity for numerical stability before sqrt
    dist_known = jnp.sqrt(jnp.maximum(0.0, dist_known_sq))  # Shape: (N, N)

    phi = multiquadric_kernel(dist_known, epsilon)  # Shape: (N, N)
    # inv_phi = jnp.linalg.inv(phi)  # Shape: (N, N)

    # Calculate squared Euclidean distances for unknown_coords vs known_coords
    # unknown_coords shape: (M, D)
    sum_unknown_sq = jnp.sum(unknown_coords**2, axis=1, keepdims=True)  # Shape: (M, 1)
    dot_new = jnp.matmul(unknown_coords, known_coords.T)  # Shape: (M, N)
    # new_dist_sq = sum_unknown_sq - 2*dot_new + sum_known_sq.T
    # sum_known_sq.T has shape (1, N)
    new_dist_sq = sum_unknown_sq - 2 * dot_new + sum_known_sq.T  # Shape: (M, N)
    new_dist = jnp.sqrt(jnp.maximum(0.0, new_dist_sq))  # Shape: (M, N)

    new_phi = multiquadric_kernel(new_dist, epsilon)  # Shape: (M, N)

    weights = jnp.linalg.solve(phi, know_values.reshape(-1))  # 求解权重
    # weights = inv_phi @ know_values.reshape(-1)
    # .flatten()
    reconstructed_values = new_phi @ weights
    return reconstructed_values


def solve_coords(dim_s, dim_t):
    s_coords, t_coords = jnp.meshgrid(jnp.arange(dim_s), jnp.arange(dim_t), indexing="ij")
    return jnp.stack([s_coords, t_coords], axis=-1)


def rbf_2x2d_interpolate_3d_vmap(iq_data: jnp.ndarray, epsilon: float) -> jnp.ndarray:
    z_dim, x_dim, t_dim = iq_data.shape

    t_mask = jnp.isnan(iq_data[0, 0, :])
    unknown_t, known_t = jnp.where(t_mask)[0], jnp.where(~t_mask)[0]

    print("开始并行处理 x-t 平面...")
    known_data = iq_data[:, :, known_t]

    all_coords_xt = solve_coords(x_dim, t_dim).astype(jnp.float32) / jnp.array(
        [x_dim, t_dim], dtype=jnp.float32
    )
    print(all_coords_xt)
    known_coords_xt = all_coords_xt[:, known_t].reshape(-1, 2)
    unknown_coords_xt = all_coords_xt[:, unknown_t].reshape(-1, 2)
    # known_value_xt_flat = jnp.transpose(iq_data[:, :, known_t], (1, 0, 2)).reshape(z_dim, -1)  # -> (z, x, t)
    # b. 预计算昂贵的RBF矩阵 (只计算一次)
    inv_phi_xt, new_phi_xt = solve_phi(known_coords_xt, unknown_coords_xt, epsilon)
    result_xt_flat = jax.vmap(rbf_2d, in_axes=(None, None, 0))(inv_phi_xt, new_phi_xt, known_data)
    # result_xt_flat = jax.lax.map(lambda x: rbf_2d(inv_phi_xt, new_phi_xt, x), known_data)
    # print(jnp.mean(result_xt_flat))
    # print(jnp.mean(result_xt_flat1))
    # print(jnp.allclose(result_xt_flat, result_xt_flat1))
    # result_xt_flat = jax.lax.map(
    #     lambda x: solve(known_coords_xt, unknown_coords_xt, x, epsilon),
    #     jnp.transpose(known_data, (1, 0, 2)),
    #     batch_size=5,
    # )
    f_rec_xt = result_xt_flat.reshape((z_dim, x_dim, -1))  # 恢复 (x,z,t)
    # f_rec_xt = jnp.transpose(result_xt_flat.reshape(z_dim, x_dim, -1), (1, 0, 2))  # 恢复 (x,z,t)
    print("x-t 平面处理完成。")

    print("开始并行处理 z-t 平面...")
    all_coords_zt = solve_coords(z_dim, t_dim).astype(jnp.float32) / jnp.array(
        [z_dim, t_dim], dtype=jnp.float32
    )
    known_coords_zt = all_coords_zt[:, known_t].reshape(-1, 2)
    unknown_coords_zt = all_coords_zt[:, unknown_t].reshape(-1, 2)

    inv_phi_zt, new_phi_zt = solve_phi(known_coords_zt, unknown_coords_zt, epsilon)
    # print("rr", rbf_2d(inv_phi_zt, new_phi_zt, known_data[:, 1, :]).block_until_ready())
    result_zt_flat = jax.lax.map(
        lambda x: rbf_2d(inv_phi_zt, new_phi_zt, x), jnp.transpose(known_data, (1, 0, 2))
    )
    f_rec_zt = jnp.transpose(result_zt_flat.reshape(x_dim, z_dim, -1), (1, 0, 2))
    # result_zt_flat = jax.vmap(rbf_2d, in_axes=(None, None, 1))(inv_phi_zt, new_phi_zt, known_data)
    # f_rec_zt = result_zt_flat.reshape(z_dim, x_dim, -1)
    print("x-t 平面处理完成。")
    # --- 4. 合并结果 ---
    print("合并结果...")
    # final_reconstruction =
    # final_reconstruction = jnp.where(jnp.isnan(iq_data), final_reconstruction, iq_data)

    # = jnp.where(jnp.isnan(iq_data), final_reconstruction, iq_data)

    return iq_data.at[:, :, unknown_t].set((f_rec_xt + f_rec_zt) / 2.0)


def test(x_dim=200, z_dim=300, t_dim=50):

    # 创建平滑变化的模拟信号
    x_range = jnp.linspace(-1, 1, x_dim)
    z_range = jnp.linspace(-1, 1, z_dim)
    t_range = jnp.linspace(0, 2 * jnp.pi, t_dim)

    # 使用meshgrid生成坐标
    xx, zz, tt = jnp.meshgrid(x_range, z_range, t_range, indexing="ij")
    center_x = 0.5 * jnp.sin(tt)
    center_z = 0.5 * jnp.cos(tt)
    ground_truth_data = jnp.exp(-((xx - center_x) ** 2 + (zz - center_z) ** 2) / 0.1)
    downsample_factor = 10
    sparse_data_np = np.array(ground_truth_data)
    for t_idx in range(t_dim):
        if t_idx % downsample_factor != 0:
            sparse_data_np[:, :, t_idx] = np.nan
    sparse_data = jnp.array(sparse_data_np)

    # --- 执行新的高性能插值函数 ---
    epsilon = 1e4
    print("--- 开始执行高性能并行版本 ---")
    reconstructed_data_v2 = rbf_2x2d_interpolate_3d_vmap(sparse_data, epsilon)
    print("\n插值完成。")
    print(f"重建后是否还存在NaN值: {jnp.isnan(reconstructed_data_v2).any()}")

    # 选择一个中间的x切片进行可视化
    slice_idx = 15

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"2x2D RBF 插值结果 (t切片索引: {slice_idx})", fontsize=16)

    # 原始数据
    im1 = axes[0].imshow(ground_truth_data[:, :, slice_idx], aspect="auto", cmap="viridis", origin="lower")
    axes[0].set_title("原始数据 (Ground Truth)")
    axes[0].set_ylabel("Z 轴")
    fig.colorbar(im1, ax=axes[0], label="信号强度")

    # 带有NaN的稀疏数据
    im2 = axes[1].imshow(
        jnp.nan_to_num(sparse_data[:, :, slice_idx]), aspect="auto", cmap="viridis", origin="lower"
    )
    axes[1].set_title(f"稀疏数据 (每 {downsample_factor} 帧保留一帧)")
    axes[1].set_ylabel("Z 轴")
    fig.colorbar(im2, ax=axes[1], label="信号强度")

    # 重建后的数据
    im3 = axes[2].imshow(
        reconstructed_data_v2[:, :, slice_idx], aspect="auto", cmap="viridis", origin="lower"
    )
    axes[2].set_title("使用 2x2D RBF 插值重建后的数据")
    axes[2].set_ylabel("Z 轴")
    axes[2].set_xlabel("时间轴 (Time)")
    fig.colorbar(im3, ax=axes[2], label="信号强度")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    # plt.savefig("rbf_interp_result.png", dpi=200)
    # plt.close()
    # print("图片已保存为 rbf_interp_result.png")


def brain(
    path: str | PathLike,
    outdir: Path,
    mode: Literal["down_fps", "cr"] = "down_fps",
    orig_fps: int = 1000,
    down_fps: int = 100,
    cr: int = 10,
):
    # mat = sio.loadmat(path)
    # iq_full = mat["IQ"].astype(jnp.complex64)
    iq_full = jnp.arange(180).reshape((3, 3, 20))
    # uf, pdata = mat.get("UF"), mat.get("PData")

    # cube = np.full_like(iq_full, np.nan, dtype=np.complex64)
    # if args.down_fps:
    #     cube[:, :, ::ds] = iq_full[:, :, ::ds]
    # else:                                     # CR 模式
    #     first_N = iq_full.shape[2] // args.cr
    #     cube[:, :, :first_N] = iq_full[:, :, :first_N]
    cube = jnp.full_like(iq_full, jnp.nan, dtype=jnp.complex64)
    match mode:
        case "down_fps":
            ds = orig_fps // down_fps
            cube = cube.at[:, :, ::ds].set(iq_full[:, :, ::ds])
        case "cr":
            first_N = iq_full.shape[2] // cr
            cube = cube.at[:, :, :first_N].set(iq_full[:, :, :first_N])
        case _:
            raise ValueError("mode must be 'down_fps' or 'cr'")
    print(cube[:, :, 0])
    t1 = time.time()
    print(f"处理 {path}，模式: {mode}，原始帧率: {orig_fps}，目标帧率: {down_fps}，CR: {cr} {t1}")
    iq_rec = rbf_2x2d_interpolate_3d_vmap(cube.real, 1e4).block_until_ready()
    # iq_rec_imag = rbf_2x2d_interpolate_3d_vmap(cube.imag, 1e4).block_until_ready()
    # iq_rec = iq_rec + 1j * iq_rec_imag
    print(f"插值完成，重建数据形状: {iq_rec.shape} {time.time() - t1}")
    # iq_rec = interp_cube(cube, args.eps, args.lam, device, args.batch)
    # out_f = os.path.join(args.out_path, fname)

    # sio.savemat(outdir / Path(path).with_suffix(".2x2rdf.mat").name, {"IQ": iq_rec, "UF": uf, "PData": pdata})


# --- 主程序：使用新的高性能函数 ---
if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Source Han Sans"]
    # jax.config.update("jax_compilation_cache_dir", "jax_cache")
    # test()
    # outdir = Path(r"data/PALA_data_InVivoRatBrain/Results-down_fps-500")
    # for d in Path(r"data/PALA_data_InVivoRatBrain/IQ").glob("*.mat"):
    #     print(f"Processing {d.name}...")
    #     brain(d, outdir, mode="down_fps", orig_fps=1000, down_fps=500)
    # d = r"data\PALA_data_InVivoRatBrain\IQ\PALA_InVivoRatBrain_001.mat"

    # outdir = Path(r"data\PALA_data_InVivoRatBrain\Results")
    # d = "data/PALA_InVivoRatBrain_001.mat"
    # outdir = Path(r"data")
    # brain(d, outdir, mode="down_fps", orig_fps=1000, down_fps=100)
    test()
