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


def multiquadric_kernel(r, epsilon):
    return jnp.sqrt(1.0 + (epsilon * r) ** 2)


@jax.jit
def rbf_2d(phi: jnp.ndarray, new_phi: jnp.ndarray, know_values: jnp.ndarray):

    weights = jax.scipy.linalg.solve(phi, know_values.reshape(-1), assume_a="sym")

    reconstructed_values = new_phi @ weights
    return reconstructed_values.reshape((know_values.shape[0], -1))


@jax.jit
def solve_phi(
    known_coords: jnp.ndarray, unknown_coords: jnp.ndarray, epsilon: float
) -> tuple[jnp.ndarray, jnp.ndarray]:
    dist_known = jnp.linalg.norm(known_coords[:, None, :] - known_coords[None, :, :], axis=-1)
    phi = multiquadric_kernel(dist_known, epsilon) + 1e-4 * jnp.eye(known_coords.shape[0])

    new_dist = jnp.linalg.norm(unknown_coords[:, None, :] - known_coords[None, :, :], axis=-1)
    new_phi = multiquadric_kernel(new_dist, epsilon)
    return phi, new_phi


def solve_coords(dim_s, dim_t):
    s_coords, t_coords = jnp.meshgrid(jnp.arange(dim_s), jnp.arange(dim_t), indexing="ij")
    return jnp.stack([s_coords, t_coords], axis=-1).astype(jnp.float32) / jnp.array(
        [dim_s, dim_t], dtype=jnp.float32
    )


@jax.jit
def rbf_2x2d_interpolate_3d_vmap_f(
    iq_data: jnp.ndarray, unknown_t: jnp.ndarray, known_t: jnp.ndarray, epsilon: float
) -> jnp.ndarray:
    z_dim, x_dim, t_dim = iq_data.shape
    dtype = iq_data.dtype

    known_data = iq_data[:, :, known_t]

    all_coords_xt = solve_coords(x_dim, t_dim)
    known_coords_xt = all_coords_xt[:, known_t].reshape(-1, 2)
    unknown_coords_xt = all_coords_xt[:, unknown_t].reshape(-1, 2)

    phi_xt, new_phi_xt = solve_phi(known_coords_xt, unknown_coords_xt, epsilon)
    f_rec_xt = jax.vmap(rbf_2d, in_axes=(None, None, 0))(phi_xt, new_phi_xt, known_data).astype(dtype)

    all_coords_zt = solve_coords(z_dim, t_dim)
    known_coords_zt = all_coords_zt[:, known_t].reshape(-1, 2)
    unknown_coords_zt = all_coords_zt[:, unknown_t].reshape(-1, 2)

    phi_zt, new_phi_zt = solve_phi(known_coords_zt, unknown_coords_zt, epsilon)

    f_rec_zt = jax.vmap(rbf_2d, in_axes=(None, None, 1), out_axes=1)(phi_zt, new_phi_zt, known_data).astype(
        dtype
    )

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

    known_data = iq_data[:, :, known_t]

    all_coords_xt = solve_coords(x_dim, t_dim)

    for x_start in range(0, x_dim, x_block_size):
        x_end = min(x_start + x_block_size, x_dim)
        current_x_slice = slice(x_start, x_end)

        block_all_coords_xt = all_coords_xt[current_x_slice, :, :]

        block_known_coords_xt = block_all_coords_xt[:, known_t, :].reshape(-1, 2)
        block_unknown_coords_xt = block_all_coords_xt[:, unknown_t, :].reshape(-1, 2)

        if block_known_coords_xt.shape[0] == 0 or block_unknown_coords_xt.shape[0] == 0:
            continue

        block_known_data_xt = known_data[:, current_x_slice, :]

        phi_xt_block, new_phi_xt_block = solve_phi(block_known_coords_xt, block_unknown_coords_xt, epsilon)

        block_rec_xt = jax.vmap(rbf_2d, in_axes=(None, None, 0))(
            phi_xt_block, new_phi_xt_block, block_known_data_xt
        ).astype(dtype)
        rec_unknowns_xt = rec_unknowns_xt.at[:, current_x_slice, :].set(block_rec_xt)

    all_coords_zt = solve_coords(x_dim, t_dim)

    for z_start in range(0, z_dim, z_block_size):
        z_end = min(z_start + z_block_size, z_dim)
        current_z_slice = slice(z_start, z_end)

        block_all_coords_zt = all_coords_zt[current_z_slice, :, :]

        block_known_coords_zt = block_all_coords_zt[:, known_t, :].reshape(-1, 2)
        block_unknown_coords_zt = block_all_coords_zt[:, unknown_t, :].reshape(-1, 2)

        if block_known_coords_zt.shape[0] == 0 or block_unknown_coords_zt.shape[0] == 0:
            continue

        block_known_data_zt = known_data[current_z_slice, :, :]

        phi_zt_block, new_phi_zt_block = solve_phi(block_known_coords_zt, block_unknown_coords_zt, epsilon)

        block_rec_zt = jax.vmap(rbf_2d, in_axes=(None, None, 1), out_axes=1)(
            phi_zt_block, new_phi_zt_block, block_known_data_zt
        ).astype(dtype)

        rec_unknowns_zt = rec_unknowns_zt.at[current_z_slice, :, :].set(block_rec_zt)

    return (rec_unknowns_xt + rec_unknowns_zt) / 2.0


def test(x_dim=40, z_dim=80, t_dim=800):
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

    result = process(ground_truth_data, mode="down_fps", ds=10)
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
    data_shape = iq_full.shape

    cube = jnp.full_like(iq_full, jnp.nan, dtype=jnp.complex64)
    match mode:
        case "down_fps":
            cube = cube.at[:, :, ::ds].set(iq_full[:, :, ::ds])
        case "cr":
            N = iq_full.shape[-1] // cr
            cube = cube.at[:, :, :N].set(iq_full[:, :, :N])
        case _:
            raise ValueError("mode must be 'down_fps' or 'cr'")

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

    sio.savemat(outdir / Path(path).with_suffix(".2x2rdf.mat").name, {"IQ": result, "UF": uf, "PData": pdata})
    del result
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
            print(f"Processed {futures[future].name} use {future.result():.2f} seconds")


def main():
    base = Path(
        r"data/Project4_2x2D RBF-based Interpolation for Ultrasound Localization Microscopy/PALA_data_InVivoRatBrain"
    )
    source = base / "IQ"
    dest = base / "test1-jax"
    dest.mkdir(exist_ok=True, parents=True)
    batch(source, dest, mode="down_fps", method="single", ds=10)


# --- 主程序：使用新的高性能函数 ---
if __name__ == "__main__":
    # plt.rcParams["font.sans-serif"] = ["Source Han Sans"]
    # jax.config.update("jax_platform_name", "cpu")
    # jax.config.update("jax_compilation_cache_dir", "jax_cache")
    jax.config.update("jax_enable_x64", True)
    main()
    # test()
