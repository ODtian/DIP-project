from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import time
from functools import partial
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import rich
import rich.progress
import scipy.io as sio
from tqdm import tqdm


# --- 复用之前的 multiquadric_kernel 函数 ---
def multiquadric_kernel(r, epsilon):
    return jnp.sqrt(1.0 + (epsilon * r) ** 2)


# --- 核心JIT函数：对单个2D切片进行插值 ---
# 这个函数现在接受预先计算好的矩阵和索引，使其与JIT兼容


@jax.jit
def rbf_2d(phi: jnp.ndarray, new_phi: jnp.ndarray, know_values: jnp.ndarray):
    # weights = phi @ know_values.reshape(-1)
    weights = jnp.linalg.solve(phi, know_values.reshape(-1))
    # print("weight", weights)
    # .flatten()
    reconstructed_values = new_phi @ weights
    return reconstructed_values.reshape((know_values.shape[0], -1))


# .reshape((know_values.shape[0], -1))


def rbf_2dd(phi: jnp.ndarray, new_phi: jnp.ndarray, know_values: jnp.ndarray):
    # weights = phi @ know_values.reshape(-1)
    weights = jnp.linalg.solve(phi, know_values.reshape(-1))
    print("val", know_values.reshape(-1))
    print("weight", weights)
    sio.savemat("data/jax.mat", {"phi": phi, "val": know_values.reshape(-1), "weight": weights})
    # print("weight", weights)
    # .flatten()
    # reconstructed_values = new_phi @ weights
    # return reconstructed_values
    return


@jax.jit
def solve_phi(
    known_coords: jnp.ndarray, unknown_coords: jnp.ndarray, epsilon: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dist_known = jnp.linalg.norm(known_coords[:, None, :] - known_coords[None, :, :], axis=-1)
    phi = multiquadric_kernel(dist_known, epsilon) + 1e-4 * jnp.eye(known_coords.shape[0])
    # inv_phi = jnp.linalg.inv(phi)  # 预计算逆矩阵

    new_dist = jnp.linalg.norm(unknown_coords[:, None, :] - known_coords[None, :, :], axis=-1)
    new_phi = multiquadric_kernel(new_dist, epsilon)
    return phi, new_phi


def solve_coords(dim_s, dim_t):
    s_coords, t_coords = jnp.meshgrid(jnp.arange(dim_s), jnp.arange(dim_t), indexing="ij")
    return jnp.stack([s_coords, t_coords], axis=-1).astype(jnp.float32) / jnp.array(
        [dim_s, dim_t], dtype=jnp.float32
    )


# @jax.jit
def rbf_2x2d_interpolate_3d_vmap_f(
    iq_data: jnp.ndarray, unknown_t: jnp.ndarray, known_t: jnp.ndarray, epsilon: float
) -> jnp.ndarray:
    z_dim, x_dim, t_dim = iq_data.shape
    dtype = iq_data.dtype

    # t_mask = jnp.isnan(iq_data[0, 0, :])
    # unknown_t, known_t = jnp.where(t_mask)[0], jnp.where(~t_mask)[0]

    # print("开始并行处理 x-t 平面...")
    known_data = iq_data[:, :, known_t]

    all_coords_xt = solve_coords(x_dim, t_dim)
    # print(all_coords_xt)
    known_coords_xt = all_coords_xt[:, known_t].reshape(-1, 2)
    unknown_coords_xt = all_coords_xt[:, unknown_t].reshape(-1, 2)
    # known_value_xt_flat = jnp.transpose(iq_data[:, :, known_t], (1, 0, 2)).reshape(z_dim, -1)  # -> (z, x, t)
    # b. 预计算昂贵的RBF矩阵 (只计算一次)
    phi_xt, new_phi_xt = solve_phi(known_coords_xt, unknown_coords_xt, epsilon)
    f_rec_xt = jax.vmap(rbf_2d, in_axes=(None, None, 0))(phi_xt, new_phi_xt, known_data).astype(dtype)
    # result_xt_flat = jax.lax.map(lambda x: rbf_2d(inv_phi_xt, new_phi_xt, x), known_data)
    # print(jnp.mean(result_xt_flat))
    # print(jnp.mean(result_xt_flat1))
    # print(jnp.allclose(result_xt_flat, result_xt_flat1))
    # result_xt_flat = jax.lax.map(
    #     lambda x: solve(known_coords_xt, unknown_coords_xt, x, epsilon),
    #     jnp.transpose(known_data, (1, 0, 2)),
    #     batch_size=5,
    # )
    # f_rec_xt = result_xt_flat  # 恢复 (x,z,t)
    # vals = known_data[2].reshape(-1)
    # print()
    # print(known_data[2].reshape(-1))
    # print()
    # w_t = jnp.linalg.solve(phi_xt, vals)
    # print(f"{w_t =}")
    # res = phi_xt @ w_t
    # print(f"{res =}")
    # print(f"{jnp.max(res - vals) =}")
    # # f_rec_xt = jnp.transpose(result_xt_flat.reshape(z_dim, x_dim, -1), (1, 0, 2))  # 恢复 (x,z,t)
    # print("x-t 平面处理完成。")

    # print("开始并行处理 z-t 平面...")
    all_coords_zt = solve_coords(z_dim, t_dim)
    known_coords_zt = all_coords_zt[:, known_t].reshape(-1, 2)
    unknown_coords_zt = all_coords_zt[:, unknown_t].reshape(-1, 2)

    phi_zt, new_phi_zt = solve_phi(known_coords_zt, unknown_coords_zt, epsilon)
    # print("rr", rbf_2d(inv_phi_zt, new_phi_zt, known_data[:, 1, :]).block_until_ready())
    # result_zt_flat = jax.lax.map(
    # lambda x: rbf_2d(inv_phi_zt, new_phi_zt, x), jnp.transpose(known_data, (1, 0, 2))
    # )
    f_rec_zt = jax.vmap(rbf_2d, in_axes=(None, None, 1), out_axes=1)(phi_zt, new_phi_zt, known_data).astype(
        dtype
    )
    # f_rec_zt = result_zt_flat.reshape((x_dim, z_dim, -1)).transpose((1, 0, 2))
    #  = result_zt_flat
    # print("x-t 平面处理完成。")
    # # --- 4. 合并结果 ---
    # print("合并结果...")
    # final_reconstruction =
    # final_reconstruction = jnp.where(jnp.isnan(iq_data), final_reconstruction, iq_data)

    # = jnp.where(jnp.isnan(iq_data), final_reconstruction, iq_data)

    return (f_rec_xt + f_rec_zt) / 2.0


def rbf_2x2d_interpolate_3d_vmap_block_f(
    iq_data: jnp.ndarray,
    unknown_t: jnp.ndarray,
    known_t: jnp.ndarray,
    epsilon: float,
    x_block_size: int = 64,
    z_block_size: int = 64,
) -> jnp.ndarray:
    z_dim, x_dim, t_dim = iq_data.shape
    dtype = iq_data.dtype

    rec_unknowns_xt = jnp.zeros((z_dim, x_dim, unknown_t.shape[0]), dtype=dtype)
    rec_unknowns_zt = jnp.zeros((z_dim, x_dim, unknown_t.shape[0]), dtype=dtype)
    # --- X-T 平面插值 (分块处理 x 轴) ---
    # print(f"开始处理 x-t 平面 (分块大小 x_block_size={x_block_size})... 设备: {gpu0}")
    known_data = iq_data[:, :, known_t]

    all_coords_xt = solve_coords(x_dim, t_dim)
    # print(all_coords_xt)

    # 全局归一化坐标 (x, t)
    # s_coords_xt_global, _t_coords_xt_global = jnp.meshgrid(
    #     jnp.arange(x_dim), jnp.arange(t_dim), indexing="ij"
    # )
    # _all_coords_xt_global_unnormalized = jnp.stack([_s_coords_xt_global, _t_coords_xt_global], axis=-1)
    # all_coords_xt_normalized_global_gpu = jax.device_put(
    #     _all_coords_xt_global_unnormalized.astype(jnp.float32) / jnp.array([x_dim, t_dim], dtype=jnp.float32),
    #     device=gpu0,
    # )

    for x_start in range(0, x_dim, x_block_size):
        x_end = min(x_start + x_block_size, x_dim)
        current_x_slice = slice(x_start, x_end)

        block_all_coords_xt = all_coords_xt[current_x_slice, :, :]

        block_known_coords_xt = block_all_coords_xt[:, known_t, :].reshape(-1, 2)
        block_unknown_coords_xt = block_all_coords_xt[:, unknown_t, :].reshape(-1, 2)

        if block_known_coords_xt.shape[0] == 0 or block_unknown_coords_xt.shape[0] == 0:
            # print(f"  跳过 x-t 块 ({x_start}-{x_end})，因为已知或未知坐标点为空。")
            continue

        block_known_data_xt = known_data[:, current_x_slice, :]

        # jit编译solve_phi，epsilon作为静态参数
        phi_xt_block, new_phi_xt_block = jax.jit(solve_phi, static_argnums=(2,))(
            block_known_coords_xt, block_unknown_coords_xt, epsilon
        )

        block_rec_xt = jax.vmap(rbf_2d, in_axes=(None, None, 0))(
            phi_xt_block, new_phi_xt_block, block_known_data_xt
        ).astype(dtype)
        # block_rec_xt = result_xt_block_flat.reshape((z_dim, x_end - x_start, len(unknown_t)))
        rec_unknowns_xt = rec_unknowns_xt.at[:, current_x_slice, :].set(block_rec_xt)

    print("x-t 平面处理完成。")

    # --- Z-T 平面插值 (分块处理 z 轴) ---
    # print(f"开始处理 z-t 平面 (分块大小 z_block_size={z_block_size})... 设备: {gpu1}")
    # 数据转置以适应z-t平面处理: (x_dim, z_dim, len(known_t))
    # known_data_zt_transposed = jnp.transpose(known_data, (1, 0, 2))

    # _s_coords_zt_global, _t_coords_zt_global = jnp.meshgrid(
    #     jnp.arange(z_dim), jnp.arange(t_dim), indexing="ij"
    # )
    # _all_coords_zt_global_unnormalized = jnp.stack([_s_coords_zt_global, _t_coords_zt_global], axis=-1)
    # all_coords_zt_normalized_global_gpu = jax.device_put(
    #     _all_coords_zt_global_unnormalized.astype(jnp.float32) / jnp.array([z_dim, t_dim], dtype=jnp.float32),
    #     device=gpu1,
    # )
    all_coords_zt = solve_coords(x_dim, t_dim)

    for z_start in range(0, z_dim, z_block_size):
        z_end = min(z_start + z_block_size, z_dim)
        current_z_slice = slice(z_start, z_end)

        block_all_coords_zt = all_coords_zt[current_z_slice, :, :]

        block_known_coords_zt = block_all_coords_zt[:, known_t, :].reshape(-1, 2)
        block_unknown_coords_zt = block_all_coords_zt[:, unknown_t, :].reshape(-1, 2)

        if block_known_coords_zt.shape[0] == 0 or block_unknown_coords_zt.shape[0] == 0:
            # print(f"  跳过 z-t 块 ({z_start}-{z_end})，因为已知或未知坐标点为空。")
            continue

        block_known_data_zt = known_data[current_z_slice, :, :]

        phi_zt_block, new_phi_zt_block = solve_phi(block_known_coords_zt, block_unknown_coords_zt, epsilon)

        block_rec_zt = jax.vmap(rbf_2d, in_axes=(None, None, 1), out_axes=1)(
            phi_zt_block, new_phi_zt_block, block_known_data_zt
        ).astype(dtype)
        # block_rec_zt_transposed = result_zt_block_flat.reshape((x_dim, z_end - z_start, len(unknown_t)))
        # block_rec_zt = jnp.transpose(block_rec_zt_transposed, (1, 0, 2))
        rec_unknowns_zt = rec_unknowns_zt.at[current_z_slice, :, :].set(block_rec_zt)

    print("z-t 平面处理完成。")

    print("合并结果...")

    # reconstructed_iq_data = iq_data.at[:, :, unknown_t].set()

    return (rec_unknowns_xt + rec_unknowns_zt) / 2.0


def test(x_dim=40, z_dim=80, t_dim=800):

    # 创建平滑变化的模拟信号
    x_range = jnp.linspace(-1, 1, x_dim)
    z_range = jnp.linspace(-1, 1, z_dim)
    t_range = jnp.linspace(0, 2 * jnp.pi, t_dim)

    # 使用meshgrid生成坐标
    xx, zz, tt = jnp.meshgrid(x_range, z_range, t_range, indexing="ij")
    center_x = 0.5 * jnp.sin(tt)
    center_z = 0.5 * jnp.cos(tt)
    # ground_truth_data = jnp.exp(-((xx - center_x) ** 2 + (zz - center_z) ** 2) / 0.1) * 100
    # ground_truth_data = np.random.random((z_dim, x_dim, t_dim)) * 100
    ground_truth_data = sio.loadmat(
        "data/Project4_2x2D RBF-based Interpolation for Ultrasound Localization Microscopy/PALA_data_InVivoRatBrain/IQ/PALA_InVivoRatBrain_022.mat"
    )["IQ"]
    print(ground_truth_data.dtype)
    result = process(ground_truth_data, mode="down_fps", ds=10)
    # downsample_factor = 10
    # sparse_data_np = np.array(ground_truth_data)
    # for t_idx in range(t_dim):
    #     if t_idx % downsample_factor != 0:
    #         sparse_data_np[:, :, t_idx] = np.nan
    # sparse_data = jnp.array(sparse_data_np)

    # # --- 执行新的高性能插值函数 ---
    # epsilon = 1e4
    # print("--- 开始执行高性能并行版本 ---")

    # t_mask = jnp.isnan(sparse_data[0, 0, :])
    # unknown_t, known_t = jnp.where(t_mask)[0], jnp.where(~t_mask)[0]
    # data_shape = sparse_data.shape

    # def pmaped():
    #     float_view = sparse_data.view(jnp.float32)
    #     iq_rec = jax.pmap(
    #         lambda x: rbf_2x2d_interpolate_3d_vmap_f(x, unknown_t, known_t, epsilon),
    #         in_axes=(3,),
    #         out_axes=3,
    #         devices=jax.devices("gpu")[:2],  # 使用前两个GPU
    #     )(float_view.reshape((*data_shape, 2)))
    #     # 确保数据类型为 float32
    #     iq_rec = iq_rec.reshape(float_view.shape).view(jnp.complex64)  # 恢复为原始形状
    #     return iq_rec.block_until_ready()

    # # reconstructed_data_v2 = rbf_2x2d_interpolate_3d_vmap_f(sparse_data, unknown_t, known_t, epsilon)
    # reconstructed_data_v2 = pmaped()
    print("\n插值完成。")
    print(f"重建后是否还存在NaN值: {jnp.isnan(result).any()}")

    # 选择一个中间的x切片进行可视化
    slice_idx = 423

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"t: {slice_idx}", fontsize=16)

    # 原始数据
    im1 = axes[0].imshow(
        jnp.abs(ground_truth_data[:, :, slice_idx]), aspect="auto", cmap="viridis", origin="lower"
    )
    axes[0].set_title("Ground Truth")
    axes[0].set_ylabel("Z")
    fig.colorbar(im1, ax=axes[0], label="amp")

    # 带有NaN的稀疏数据
    # im2 = axes[1].imshow(
    #     jnp.abs(jnp.nan_to_num(sparse_data[:, :, slice_idx])),
    #     aspect="auto",
    #     cmap="viridis",
    #     # origin="lower",
    # )
    # axes[1].set_title(f"sparse data {downsample_factor} ")
    # axes[1].set_ylabel("Z")
    # fig.colorbar(im2, ax=axes[1], label="amp")

    # 重建后的数据
    im3 = axes[2].imshow(jnp.abs(result[:, :, slice_idx]), aspect="auto", cmap="viridis", origin="lower")
    axes[2].set_title("")
    axes[2].set_ylabel("Z")
    axes[2].set_xlabel("Time")
    fig.colorbar(im3, ax=axes[2], label="amp")

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.show()
    plt.savefig("rbf_interp_result2.png", dpi=200)
    # plt.close()
    # print("图片已保存为 rbf_interp_result.png")


def brain(
    path: str | PathLike,
    outdir: Path,
    mode: Literal["down_fps", "cr"] = "down_fps",
    method: Literal["pmap", "single"] = "pmap",
    blocking: bool = False,
    block_x: int = 64,
    block_z: int = 64,
    orig_fps: int = 1000,
    down_fps: int = 100,
    cr: int = 10,
):
    print(f"{orig_fps=} {down_fps=}")
    mat = sio.loadmat(path, mat_dtype=True)
    iq_full = jnp.array(sio.loadmat(path)["IQ"]).astype(jnp.complex64)  # 确保数据类型为复数
    # print(iq_full.dtype)
    # .astype(jnp.complex64)  # 确保数据类型为复数
    data_shape = iq_full.shape
    # iq_full = jnp.arange(180).reshape((3, 3, 20))
    uf, pdata = mat.get("UF"), mat.get("PData")

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
            first_N = iq_full.shape[-1] // cr
            cube = cube.at[:, :, :first_N].set(iq_full[:, :, :first_N])
        case _:
            raise ValueError("mode must be 'down_fps' or 'cr'")
    t1 = time.time()
    print(f"处理 {path}，模式: {mode}，原始帧率: {orig_fps}，目标帧率: {down_fps}，CR: {cr} {t1}")
    # iq_rec = rbf_2x2d_interpolate_3d_vmap(cube.real[50:51, 50:51, :], 1e4).block_until_ready()
    # iq_rec = rbf_2x2d_interpolate_3d_vmap(cube.real, 1e4).block_until_ready()
    # iq_rec = rbf_2x2d_interpolate_3d_vmap_block(float_view, 1e4)

    # iq_rec = jax.lax.map(
    #     lambda x: rbf_2x2d_interpolate_3d_vmap(x, 1e4),
    #     jnp.moveaxis(float_view.reshape((*data_shape, 2)), -1, 0),
    # )
    t_mask = jnp.isnan(cube[0, 0, :])
    unknown_t, known_t = jnp.where(t_mask)[0], jnp.where(~t_mask)[0]

    if blocking:
        func = partial(rbf_2x2d_interpolate_3d_vmap_block_f, x_block_size=block_x, z_block_size=block_z)
    else:
        func = rbf_2x2d_interpolate_3d_vmap_f

    def pmaped():
        float_view = cube.view(jnp.float32)
        vector_view = float_view.reshape((*data_shape, 2))  # 将复数数据转换为实数向量
        iq_rec = jax.pmap(
            lambda x: rbf_2x2d_interpolate_3d_vmap_f(x, unknown_t, known_t, 1e4),
            in_axes=(3,),
            out_axes=3,
            devices=jax.devices("gpu")[:2],  # 使用前两个GPU
        )(vector_view)
        # 确保数据类型为 float32
        return (
            vector_view.at[:, :, unknown_t]
            .set(iq_rec)
            .reshape(float_view.shape)
            .view(jnp.complex64)
            .block_until_ready()
        )

    def single():
        iq_rec = rbf_2x2d_interpolate_3d_vmap_f(cube.real, unknown_t, known_t, 1e4)
        return iq_full.at[:, :, unknown_t].set(iq_rec + 0j)

    match method:
        case "pmap":
            result = pmaped()
        case "single":
            result = single()
        case _:
            raise ValueError("method must be 'pmap' or 'single'")
    # iq_rec = iq_full
    # iq_rec = single()
    # iq_rec = jnp.moveaxis(iq_rec, 0, -1).reshape(float_view.shape).view(jnp.complex64)  # 恢复为复数类型
    # iq_rec_real = jax.device_get(iq_rec_real)

    #
    # iq_rec_image = jax.device_get(iq_rec_image)
    # iq_rec_imag = rbf_2x2d_interpolate_3d_vmap(cube.imag, 1e4).block_until_ready()
    # iq_rec = iq_rec + 1j * iq_rec_imag
    print(f"插值完成，重建数据形状: {result.shape} {time.time() - t1}")
    # iq_rec = interp_cube(cube, args.eps, args.lam, device, args.batch)
    # out_f = os.path.join(args.out_path, fname)

    sio.savemat(outdir / Path(path).with_suffix(".2x2rdf.mat").name, {"IQ": result, "UF": uf, "PData": pdata})


def process(
    iq_full: jnp.ndarray,
    mode: Literal["down_fps", "cr"] = "down_fps",
    method: Literal["pmap", "single"] = "pmap",
    blocking: bool = False,
    block_x: int = 64,
    block_z: int = 64,
    ds: int = 10,
    cr: int = 10,
):
    # print(iq_full.dtype)
    # .astype(jnp.complex64)  # 确保数据类型为复数
    data_shape = iq_full.shape
    # iq_full = jnp.arange(180).reshape((3, 3, 20))

    # cube = np.full_like(iq_full, np.nan, dtype=np.complex64)
    # if args.down_fps:
    #     cube[:, :, ::ds] = iq_full[:, :, ::ds]
    # else:                                     # CR 模式
    #     first_N = iq_full.shape[2] // args.cr
    #     cube[:, :, :first_N] = iq_full[:, :, :first_N]
    cube = jnp.full_like(iq_full, jnp.nan, dtype=jnp.complex64)
    match mode:
        case "down_fps":
            cube = cube.at[:, :, ::ds].set(iq_full[:, :, ::ds])
        case "cr":
            N = iq_full.shape[-1] // cr
            cube = cube.at[:, :, :N].set(iq_full[:, :, :N])
        case _:
            raise ValueError("mode must be 'down_fps' or 'cr'")

    # iq_rec = rbf_2x2d_interpolate_3d_vmap(cube.real[50:51, 50:51, :], 1e4).block_until_ready()
    # iq_rec = rbf_2x2d_interpolate_3d_vmap(cube.real, 1e4).block_until_ready()
    # iq_rec = rbf_2x2d_interpolate_3d_vmap_block(float_view, 1e4)

    # iq_rec = jax.lax.map(
    #     lambda x: rbf_2x2d_interpolate_3d_vmap(x, 1e4),
    #     jnp.moveaxis(float_view.reshape((*data_shape, 2)), -1, 0),
    # )
    t_mask = jnp.isnan(cube[0, 0, :])
    unknown_t, known_t = jnp.where(t_mask)[0], jnp.where(~t_mask)[0]

    if blocking:
        func = partial(rbf_2x2d_interpolate_3d_vmap_block_f, x_block_size=block_x, z_block_size=block_z)
    else:
        func = rbf_2x2d_interpolate_3d_vmap_f

    def pmaped():
        float_view = cube.view(jnp.float32)
        vector_view = float_view.reshape((*data_shape, 2))  # 将复数数据转换为实数向量
        iq_rec = jax.pmap(
            lambda x: func(x, unknown_t, known_t, 1e4),
            in_axes=(3,),
            out_axes=3,
            devices=jax.devices("gpu")[:2],  # 使用前两个GPU
        )(vector_view)
        # 确保数据类型为 float32
        return vector_view.at[:, :, unknown_t].set(iq_rec).reshape(float_view.shape).view(jnp.complex64)

    def single():
        iq_rec = func(cube.real, unknown_t, known_t, 1e4)
        return iq_full.at[:, :, unknown_t].set(iq_rec + 0j)

    match method:
        case "pmap":
            result = pmaped()
        case "single":
            result = single()
        case _:
            raise ValueError("method must be 'pmap' or 'single'")
    return result
    # iq_rec = iq_full
    # iq_rec = single()
    # iq_rec = jnp.moveaxis(iq_rec, 0, -1).reshape(float_view.shape).view(jnp.complex64)  # 恢复为复数类型
    # iq_rec_real = jax.device_get(iq_rec_real)

    #
    # iq_rec_image = jax.device_get(iq_rec_image)
    # iq_rec_imag = rbf_2x2d_interpolate_3d_vmap(cube.imag, 1e4).block_until_ready()
    # iq_rec = iq_rec + 1j * iq_rec_imag


def process_mat(
    path: str | PathLike,
    outdir: Path,
    device: Any,
    mode: Literal["down_fps", "cr"] = "down_fps",
    method: Literal["pmap", "single"] = "pmap",
    blocking: bool = False,
    block_x: int = 64,
    block_z: int = 64,
    ds: int = 10,
    cr: int = 10,
):
    mat = sio.loadmat(path, mat_dtype=True)
    iq_full = jnp.array(sio.loadmat(path)["IQ"], device=device).astype(jnp.complex64)  # 确保数据类型为复数

    uf, pdata = mat.get("UF"), mat.get("PData")

    t1 = time.time()

    result = process(
        iq_full,
        mode=mode,
        method=method,
        blocking=blocking,
        block_x=block_x,
        block_z=block_z,
        ds=ds,
        cr=cr,
    )
    # iq_rec = iq_full
    # iq_rec = single()
    # iq_rec = jnp.moveaxis(iq_rec, 0, -1).reshape(float_view.shape).view(jnp.complex64)  # 恢复为复数类型
    # iq_rec_real = jax.device_get(iq_rec_real)

    #
    # iq_rec_image = jax.device_get(iq_rec_image)
    # iq_rec_imag = rbf_2x2d_interpolate_3d_vmap(cube.imag, 1e4).block_until_ready()
    # iq_rec = iq_rec + 1j * iq_rec_imag

    # iq_rec = interp_cube(cube, args.eps, args.lam, device, args.batch)
    # out_f = os.path.join(args.out_path, fname)

    sio.savemat(outdir / Path(path).with_suffix(".2x2rdf.mat").name, {"IQ": result, "UF": uf, "PData": pdata})

    return time.time() - t1


def batch(
    source: str | PathLike,
    dest: str | PathLike,
    mode: Literal["down_fps", "cr"] = "down_fps",
    method: Literal["pmap", "single"] = "pmap",
    blocking: bool = False,
    block_x: int = 64,
    block_z: int = 64,
    ds: int = 10,
    cr: int = 10,
):

    # items = tuple(sorted((datadir / "IQ").glob("*.mat")))
    items = tuple(sorted(Path(source).glob("*.mat")))
    outdir = Path(dest)

    devices = jax.devices("gpu")
    num_devices = len(devices)
    with ThreadPoolExecutor(max_workers=num_devices) as executor:
        futures: dict[Future, Path] = {}
        for i, d in enumerate(items):
            future = executor.submit(
                process_mat,
                d,
                outdir,
                devices[i % num_devices],
                mode=mode,
                method=method,
                blocking=blocking,
                block_x=block_x,
                block_z=block_z,
                ds=ds,
                cr=cr,
            )
            futures[future] = d

        for future in rich.progress.track(as_completed(futures), total=len(futures)):
            # for d in rich.progress.track(items[:]):
            print(f"Processed {futures[future].name} use {future.result():.2f} seconds")
            # brain(d, outdir, mode="down_fps", orig_fps=1000, down_fps=100, method="single", blocking=True)
            # break


def main2():
    d = [
        # d = r"data\PALA_data_InVivoRatBrain\IQ\PALA_InVivoRatBrain_001.mat"
        "data/test/sparse_data.mat",
        "data/PALA_InVivoRatBrain_001.mat",
    ]

    outdir = Path(r"data/result")
    brain(d[1], outdir, mode="down_fps", orig_fps=1000, down_fps=100)


def main():
    d = [
        # d = r"data\PALA_data_InVivoRatBrain\IQ\PALA_InVivoRatBrain_001.mat"
        "data/test/sparse_data.mat",
        "data/PALA_InVivoRatBrain_001.mat",
    ]
    base = Path(
        r"data/Project4_2x2D RBF-based Interpolation for Ultrasound Localization Microscopy/PALA_data_InVivoRatBrain"
    )
    source = base / "IQ"
    dest = base / "down_fps-100-jax"
    dest.mkdir(exist_ok=True, parents=True)
    # outdir = Path(r"data/result")
    batch(source, dest, mode="down_fps", method="single", ds=10)


# --- 主程序：使用新的高性能函数 ---
if __name__ == "__main__":
    # plt.rcParams["font.sans-serif"] = ["Source Han Sans"]
    # jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_compilation_cache_dir", "jax_cache")
    jax.config.update("jax_enable_x64", True)
    # jax.config.update("jax_numpy_dtype_promotion", "strict")
    main()
    # test()
