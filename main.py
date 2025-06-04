# from functools import partial
# import jax
# import jax.numpy as jnp
# import numpy as np
# import matplotlib.pyplot as plt

# # JAX 配置，如果需要，可以使用双精度
# # from jax import config
# # config.update("jax_enable_x64", True)


# def multiquadric_kernel(r, epsilon):
#     """
#     计算多象限 (Multiquadric) RBF 核函数。
#     根据论文中的公式：φ(r) = (1 + (εr)^2)^(1/2) [cite: 72]

#     参数:
#         r (jax.Array): 欧几里得距离。
#         epsilon (float): 控制函数形状的形状参数。

#     返回:
#         jax.Array: RBF核的计算结果。
#     """
#     return jnp.sqrt(1 + (epsilon * r) ** 2)


# # @jax.jit
# # def rbf_interpolate_2d(grid_2d: jnp.ndarray, epsilon: float):
# def rbf_interpolate_2d(
#     known_coords: jnp.ndarray, known_values: jnp.ndarray, unknown_coords: jnp.ndarray, epsilon: float
# ):
#     """
#     使用RBF对包含NaN值的2D网格进行插值。

#     参数:
#         grid_2d (jax.Array): 形状为 (space, time) 的2D输入网格，其中缺失值由NaN表示。
#         epsilon (float): RBF核的形状参数。

#     返回:
#         jax.Array: 完成插值的2D网格。
#     """

#     # 如果没有已知点或没有需要插值的点，则直接返回
#     # if known_coords.shape[0] == 0 or unknown_coords.shape[0] == 0:
#     #     return grid_2d

#     # 2. 构建模型：求解权重 λ (lambda)
#     # 线性系统为 ΦΛ = F [cite: 75]
#     # 计算已知点之间的距离矩阵
#     dist_matrix_known = jnp.linalg.norm(known_coords[:, None, :] - known_coords[None, :, :], axis=-1)

#     # 计算RBF核矩阵 Φ
#     phi_matrix = multiquadric_kernel(dist_matrix_known, epsilon)

#     # 求解权重：Λ = Φ⁻¹F。使用 jnp.linalg.solve 更稳定 [cite: 76]
#     weights = jnp.linalg.solve(phi_matrix, known_values)

#     # 3. 利用模型：重建未知值
#     # 计算未知点与已知点之间的距离矩阵 [cite: 79]
#     dist_matrix_unknown_to_known = jnp.linalg.norm(
#         unknown_coords[:, None, :] - known_coords[None, :, :], axis=-1
#     )

#     # 构建新的估值矩阵 Φ_new [cite: 78]
#     phi_new_matrix = multiquadric_kernel(dist_matrix_unknown_to_known, epsilon)

#     # 计算重建值：F_rec = Φ_new * Λ [cite: 80]
#     reconstructed_values = phi_new_matrix @ weights

#     return reconstructed_values
#     # # 4. 将重建值填充回网格
#     # reconstructed_grid = grid_2d.flatten().at[unknown_mask].set(reconstructed_values)

#     # return reconstructed_grid.reshape(space_dim, time_dim)


# def split_unknown(grid_2d: jnp.ndarray):
#     """
#     分离已知和未知点的坐标和数值。
#     该函数将输入的2D网格分为已知点和未知点（NaN值）。

#     参数:
#         grid_2d (jax.Array): 形状为 (space, time) 的2D输入网格，其中缺失值由NaN表示。

#     返回:
#         tuple: 包含已知点坐标、已知点数值、未知点坐标的元组。
#     """
#     # 1. 识别已知点和未知点
#     space_dim, time_dim = grid_2d.shape
#     s_coords, t_coords = jnp.meshgrid(jnp.arange(space_dim), jnp.arange(time_dim), indexing="ij")

#     all_coords = jnp.stack([s_coords.flatten(), t_coords.flatten()], axis=1)
#     all_values = grid_2d.flatten()

#     known_mask = ~jnp.isnan(all_values)
#     unknown_mask = jnp.isnan(all_values)

#     known_coords = all_coords[known_mask]
#     known_values = all_values[known_mask]
#     unknown_coords = all_coords[unknown_mask]

#     return known_coords, known_values, unknown_coords, unknown_mask


# # def split_unknown_3d(grid_3d: jnp.ndarray):

# #     # 1. 识别已知点和未知点
# #     x_dim, z_dim, t_dim = grid_3d.shape
# #     x_coords, z_coords, t_coords = jnp.meshgrid(
# #         jnp.arange(x_dim), jnp.arange(z_dim), jnp.arange(t_dim), indexing="ijk"
# #     )

# #     all_coords = jnp.stack([x_coords, z_coords, t_coords], axis=1)
# #     all_values = grid_3d

# #     known_mask = ~jnp.isnan(all_values)
# #     unknown_mask = jnp.isnan(all_values)

# #     known_coords = all_coords[known_mask]
# #     known_values = all_values[known_mask]
# #     unknown_coords = all_coords[unknown_mask]

# #     return known_coords, known_values, unknown_coords, unknown_mask


# def rbf_2x2d_interpolate_3d(iq_data: jnp.ndarray, epsilon: float) -> jnp.ndarray:
#     """
#     执行双向 (2x2D) RBF插值。
#     该函数分别在 x-t 和 z-t 平面上进行插值，然后合并结果 [cite: 86]。

#     参数:
#         iq_data (jax.Array): 形状为 (x, z, t) 的3D IQ数据，缺失值由NaN表示。
#         epsilon (float): RBF核的形状参数。

#     返回:
#         jax.Array: 完成插值的3D数据。
#     """
#     x_dim, z_dim, t_dim = iq_data.shape

#     # 初始化用于存储两个方向插值结果的数组
#     f_rec_xt = jnp.zeros_like(iq_data)
#     f_rec_zt = jnp.zeros_like(iq_data)

#     print("开始在 x-t 平面上进行插值...")
#     # --- 1. 沿 x-t 平面插值 (对每个z深度) ---
#     for z_idx in range(z_dim):
#         # xt_slice =
#         xt_slice = iq_data[:, z_idx, :]  # 形状 (x, t)
#         known_coords, known_values, unknown_coords, unknown_mask = split_unknown(xt_slice)
#         interpolated_slice = rbf_interpolate_2d(known_coords, known_values, unknown_coords, epsilon)
#         # # 保持未知点为NaN
#         f_rec_xt = f_rec_xt.at[:, z_idx, :].set(
#             xt_slice.flatten().at[unknown_mask].set(interpolated_slice).reshape(x_dim, t_dim)
#         )
#     print("x-t 平面插值完成。")

#     print("开始在 z-t 平面上进行插值...")
#     # --- 2. 沿 z-t 平面插值 (对每个x位置) ---
#     for x_idx in range(x_dim):
#         zt_slice = iq_data[x_idx, :, :]  # 形状 (z, t)
#         interpolated_slice = rbf_interpolate_2d(zt_slice, epsilon)
#         f_rec_zt = f_rec_zt.at[x_idx, :, :].set(interpolated_slice)
#     print("z-t 平面插值完成。")

#     # --- 3. 合并结果 ---
#     # 论文中提到，如果没有先验知识，可以将结果相加 [cite: 94]。
#     # 对于密度图，论文建议取平均值以避免不必要的错误 [cite: 102]。
#     # 此处使用平均值，这是一种更稳健的组合方式。
#     print("合并两个方向的插值结果...")
#     final_reconstruction = (f_rec_xt + f_rec_zt) / 2.0

#     # 对于原始数据中已知的部分，直接使用原始值
#     known_mask = ~jnp.isnan(iq_data)
#     final_reconstruction = jnp.where(known_mask, iq_data, final_reconstruction)

#     return final_reconstruction


# # --- 核心JIT函数：对单个2D切片进行插值 ---
# # 这个函数现在接受预先计算好的矩阵和索引，使其与JIT兼容
# @partial(jax.jit, static_argnames=["slice_shape"])
# def _rbf_interpolate_2d_jit(
#     flat_slice: jnp.ndarray,
#     known_indices_flat: jnp.ndarray,
#     unknown_indices_flat: jnp.ndarray,
#     phi_matrix_inv: jnp.ndarray,  # 传入预计算的逆矩阵
#     phi_new_matrix: jnp.ndarray,
#     slice_shape: tuple[int, int],  # 传入原始2D切片的形状 (x_dim, t_dim
# ):
#     """
#     一个JIT兼容的函数，对单个2D切片进行插值。

#     参数:
#         flat_slice (jax.Array): 展平后的一维数据切片。
#         known_indices_flat (jax.Array): 展平后已知点的整数索引。
#         unknown_indices_flat (jax.Array): 展平后未知点的整数索引。
#         phi_matrix_inv (jax.Array): 预计算的RBF核矩阵的逆。
#         phi_new_matrix (jax.Array): 预计算的新估值矩阵。
#         slice_shape (tuple): 原始2D切片的形状。
#     """
#     # 1. 使用整数索引gather已知值 (JIT兼容)
#     known_values = flat_slice.take(known_indices_flat)

#     # 2. 计算权重 (现在是矩阵-向量乘法，非常快)
#     # Λ = Φ⁻¹ F
#     weights = phi_matrix_inv @ known_values

#     # 3. 重建未知值
#     # F_rec = Φ_new @ Λ
#     reconstructed_values = phi_new_matrix @ weights

#     # 4. 将已知值和重建值scatter回一个完整数组 (JIT兼容)
#     result_flat = jnp.zeros_like(flat_slice)
#     result_flat = result_flat.at[known_indices_flat].set(known_values)
#     result_flat = result_flat.at[unknown_indices_flat].set(reconstructed_values)

#     return result_flat.reshape(slice_shape)


# # --- 新的主函数：高性能并行版本 ---
# def rbf_2x2d_interpolate_3d_v2(iq_data, epsilon):
#     """
#     高性能并行版的2x2D RBF插值。
#     使用 vmap 代替 for 循环，并预计算矩阵。
#     """
#     x_dim, z_dim, t_dim = iq_data.shape

#     # --- 1. 预计算时间索引 ---
#     # 假设所有切片的时间模式相同
#     time_mask = ~jnp.isnan(iq_data[0, 0, :])
#     known_time_indices = jnp.where(time_mask)[0]
#     unknown_time_indices = jnp.where(~time_mask)[0]

#     # --- 2. 处理 x-t 平面 (并行处理所有z切片) ---
#     print("开始并行处理 x-t 平面...")
#     # a. 预计算 x-t 平面的几何信息
#     xt_shape = (x_dim, t_dim)
#     s_coords, t_coords = jnp.meshgrid(jnp.arange(x_dim), jnp.arange(t_dim), indexing="ij")
#     all_coords_xt_flat = jnp.stack([s_coords.flatten(), t_coords.flatten()], axis=1)

#     time_mask_2d_xt = jnp.tile(time_mask, x_dim)
#     known_indices_xt_flat = jnp.where(time_mask_2d_xt)[0]
#     unknown_indices_xt_flat = jnp.where(~time_mask_2d_xt)[0]

#     known_coords_xt = all_coords_xt_flat[known_indices_xt_flat]
#     unknown_coords_xt = all_coords_xt_flat[unknown_indices_xt_flat]

#     # b. 预计算昂贵的RBF矩阵 (只计算一次)
#     dist_known_xt = jnp.linalg.norm(known_coords_xt[:, None, :] - known_coords_xt[None, :, :], axis=-1)
#     phi_matrix_xt = multiquadric_kernel(dist_known_xt, epsilon)
#     phi_matrix_xt_inv = jnp.linalg.inv(phi_matrix_xt)  # 预计算逆矩阵

#     dist_new_xt = jnp.linalg.norm(unknown_coords_xt[:, None, :] - known_coords_xt[None, :, :], axis=-1)
#     phi_new_matrix_xt = multiquadric_kernel(dist_new_xt, epsilon)

#     # c. 准备数据并vmap
#     data_for_xt = jnp.transpose(iq_data, (1, 0, 2))  # -> (z, x, t)
#     data_for_xt_flat = data_for_xt.reshape(z_dim, -1)  # -> (z, x*t)

#     # 使用vmap将函数应用到每个z切片上
#     vmapped_interp_xt = jax.vmap(_rbf_interpolate_2d_jit, in_axes=(0, None, None, None, None, None))

#     # 执行并行计算
#     result_xt_flat = vmapped_interp_xt(
#         data_for_xt_flat,
#         known_indices_xt_flat,
#         unknown_indices_xt_flat,
#         phi_matrix_xt_inv,
#         phi_new_matrix_xt,
#         xt_shape,
#     )
#     f_rec_xt = jnp.transpose(result_xt_flat.reshape(z_dim, x_dim, t_dim), (1, 0, 2))  # 恢复 (x,z,t)
#     print("x-t 平面处理完成。")

#     # --- 3. 处理 z-t 平面 (并行处理所有x切片) ---
#     print("开始并行处理 z-t 平面...")
#     # a. 预计算 z-t 平面的几何信息
#     zt_shape = (z_dim, t_dim)
#     s_coords, t_coords = jnp.meshgrid(jnp.arange(z_dim), jnp.arange(t_dim), indexing="ij")
#     all_coords_zt_flat = jnp.stack([s_coords.flatten(), t_coords.flatten()], axis=1)

#     time_mask_2d_zt = jnp.tile(time_mask, z_dim)
#     known_indices_zt_flat = jnp.where(time_mask_2d_zt)[0]
#     unknown_indices_zt_flat = jnp.where(~time_mask_2d_zt)[0]

#     known_coords_zt = all_coords_zt_flat[known_indices_zt_flat]
#     unknown_coords_zt = all_coords_zt_flat[unknown_indices_zt_flat]

#     # b. 预计算RBF矩阵
#     dist_known_zt = jnp.linalg.norm(known_coords_zt[:, None, :] - known_coords_zt[None, :, :], axis=-1)
#     phi_matrix_zt = multiquadric_kernel(dist_known_zt, epsilon)
#     phi_matrix_zt_inv = jnp.linalg.inv(phi_matrix_zt)

#     dist_new_zt = jnp.linalg.norm(unknown_coords_zt[:, None, :] - known_coords_zt[None, :, :], axis=-1)
#     phi_new_matrix_zt = multiquadric_kernel(dist_new_zt, epsilon)

#     # c. 准备数据并vmap
#     data_for_zt_flat = iq_data.reshape(x_dim, -1)  # -> (x, z*t)
#     vmapped_interp_zt = jax.vmap(_rbf_interpolate_2d_jit, in_axes=(0, None, None, None, None, None))
#     result_zt_flat = vmapped_interp_zt(
#         data_for_zt_flat,
#         known_indices_zt_flat,
#         unknown_indices_zt_flat,
#         phi_matrix_zt_inv,
#         phi_new_matrix_zt,
#         zt_shape,
#     )
#     f_rec_zt = result_zt_flat.reshape(x_dim, z_dim, t_dim)
#     print("z-t 平面处理完成。")

#     # --- 4. 合并结果 ---
#     print("合并结果...")
#     final_reconstruction = (f_rec_xt + f_rec_zt) / 2.0
#     final_reconstruction = jnp.where(jnp.isnan(iq_data), final_reconstruction, iq_data)

#     return final_reconstruction


# if __name__ == "__main__":
#     plt.rcParams["font.sans-serif"] = ["Source Han Sans"]
#     plt.rcParams["axes.unicode_minus"] = False
#     # mpl.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号
#     # --- 1. 创建模拟数据 ---
#     # 数据维度：(x_dim, z_dim, time_dim)
#     x_dim, z_dim, t_dim = 20, 30, 50

#     # 创建平滑变化的模拟信号
#     x_range = jnp.linspace(-1, 1, x_dim)
#     z_range = jnp.linspace(-1, 1, z_dim)
#     t_range = jnp.linspace(0, 2 * jnp.pi, t_dim)

#     # 使用meshgrid生成坐标
#     xx, zz, tt = jnp.meshgrid(x_range, z_range, t_range, indexing="ij")

#     # 一个移动的高斯波包作为模拟信号
#     center_x = 0.5 * jnp.sin(tt)
#     center_z = 0.5 * jnp.cos(tt)

#     ground_truth_data = jnp.exp(-((xx - center_x) ** 2 + (zz - center_z) ** 2) / 0.1)

#     print(f"创建的原始数据形状: {ground_truth_data.shape}")

#     # --- 2. 模拟低帧率/缺失数据 ---
#     # 模拟“策略1”，即降低帧率采集 [cite: 113]
#     # 我们保留每5帧中的一帧，其余设为NaN
#     downsample_factor = 5
#     sparse_data = np.array(ground_truth_data)  # 转换为numpy以便于修改

#     for t_idx in range(t_dim):
#         if t_idx % downsample_factor != 0:
#             sparse_data[:, :, t_idx] = np.nan

#     sparse_data = jnp.array(sparse_data)  # 转换回JAX数组

#     # --- 3. 执行 2x2D RBF 插值 ---
#     # 论文中提到 epsilon=10^4 用于减少不稳定性 [cite: 85]
#     epsilon = 1e4

#     reconstructed_data = rbf_2x2d_interpolate_3d(sparse_data, epsilon)

#     print("\n插值完成。")
#     print(f"重建后的数据形状: {reconstructed_data.shape}")
#     print(f"重建后是否还存在NaN值: {jnp.isnan(reconstructed_data).any()}")

#     # --- 4. 可视化结果 ---
#     # 选择一个中间的x切片进行可视化
#     slice_idx = x_dim // 2

#     fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
#     fig.suptitle(f"2x2D RBF 插值结果 (x切片索引: {slice_idx})", fontsize=16)

#     # 原始数据
#     im1 = axes[0].imshow(ground_truth_data[slice_idx, :, :], aspect="auto", cmap="viridis", origin="lower")
#     axes[0].set_title("原始数据 (Ground Truth)")
#     axes[0].set_ylabel("Z 轴")
#     fig.colorbar(im1, ax=axes[0], label="信号强度")

#     # 带有NaN的稀疏数据
#     im2 = axes[1].imshow(
#         jnp.nan_to_num(sparse_data[slice_idx, :, :]), aspect="auto", cmap="viridis", origin="lower"
#     )
#     axes[1].set_title(f"稀疏数据 (每 {downsample_factor} 帧保留一帧)")
#     axes[1].set_ylabel("Z 轴")
#     fig.colorbar(im2, ax=axes[1], label="信号强度")

#     # 重建后的数据
#     im3 = axes[2].imshow(reconstructed_data[slice_idx, :, :], aspect="auto", cmap="viridis", origin="lower")
#     axes[2].set_title("使用 2x2D RBF 插值重建后的数据")
#     axes[2].set_ylabel("Z 轴")
#     axes[2].set_xlabel("时间轴 (Time)")
#     fig.colorbar(im3, ax=axes[2], label="信号强度")

#     plt.tight_layout(rect=(0, 0, 1, 0.96))
#     plt.show()


import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial


# --- 复用之前的 multiquadric_kernel 函数 ---
def multiquadric_kernel(r, epsilon):
    return jnp.sqrt(1 + (epsilon * r) ** 2)


# --- 核心JIT函数：对单个2D切片进行插值 ---
# 这个函数现在接受预先计算好的矩阵和索引，使其与JIT兼容
@partial(jax.jit, static_argnames=["slice_shape"])
def _rbf_interpolate_2d_jit(
    flat_slice,
    known_indices_flat,
    unknown_indices_flat,
    phi_matrix_inv,  # 传入预计算的逆矩阵
    phi_new_matrix,
    slice_shape,
):
    """
    一个JIT兼容的函数，对单个2D切片进行插值。

    参数:
        flat_slice (jax.Array): 展平后的一维数据切片。
        known_indices_flat (jax.Array): 展平后已知点的整数索引。
        unknown_indices_flat (jax.Array): 展平后未知点的整数索引。
        phi_matrix_inv (jax.Array): 预计算的RBF核矩阵的逆。
        phi_new_matrix (jax.Array): 预计算的新估值矩阵。
        slice_shape (tuple): 原始2D切片的形状。
    """
    # 1. 使用整数索引gather已知值 (JIT兼容)
    known_values = flat_slice.take(known_indices_flat)

    # 2. 计算权重 (现在是矩阵-向量乘法，非常快)
    # Λ = Φ⁻¹ F
    weights = phi_matrix_inv @ known_values

    # 3. 重建未知值
    # F_rec = Φ_new @ Λ
    reconstructed_values = phi_new_matrix @ weights

    # 4. 将已知值和重建值scatter回一个完整数组 (JIT兼容)
    result_flat = jnp.zeros_like(flat_slice)
    result_flat = result_flat.at[known_indices_flat].set(known_values)
    result_flat = result_flat.at[unknown_indices_flat].set(reconstructed_values)

    return result_flat.reshape(slice_shape)


# --- 新的主函数：高性能并行版本 ---
def rbf_2x2d_interpolate_3d_v2(iq_data, epsilon):
    """
    高性能并行版的2x2D RBF插值。
    使用 vmap 代替 for 循环，并预计算矩阵。
    """
    x_dim, z_dim, t_dim = iq_data.shape

    # --- 1. 预计算时间索引 ---
    # 假设所有切片的时间模式相同
    time_mask = ~jnp.isnan(iq_data[0, 0, :])
    known_time_indices = jnp.where(time_mask)[0]
    unknown_time_indices = jnp.where(~time_mask)[0]

    # --- 2. 处理 x-t 平面 (并行处理所有z切片) ---
    print("开始并行处理 x-t 平面...")
    # a. 预计算 x-t 平面的几何信息
    xt_shape = (x_dim, t_dim)
    s_coords, t_coords = jnp.meshgrid(jnp.arange(x_dim), jnp.arange(t_dim), indexing="ij")
    all_coords_xt_flat = jnp.stack([s_coords.flatten(), t_coords.flatten()], axis=1)

    time_mask_2d_xt = jnp.tile(time_mask, x_dim)
    known_indices_xt_flat = jnp.where(time_mask_2d_xt)[0]
    unknown_indices_xt_flat = jnp.where(~time_mask_2d_xt)[0]

    known_coords_xt = all_coords_xt_flat[known_indices_xt_flat]
    unknown_coords_xt = all_coords_xt_flat[unknown_indices_xt_flat]

    # b. 预计算昂贵的RBF矩阵 (只计算一次)
    dist_known_xt = jnp.linalg.norm(known_coords_xt[:, None, :] - known_coords_xt[None, :, :], axis=-1)
    phi_matrix_xt = multiquadric_kernel(dist_known_xt, epsilon)
    phi_matrix_xt_inv = jnp.linalg.inv(phi_matrix_xt)  # 预计算逆矩阵

    dist_new_xt = jnp.linalg.norm(unknown_coords_xt[:, None, :] - known_coords_xt[None, :, :], axis=-1)
    phi_new_matrix_xt = multiquadric_kernel(dist_new_xt, epsilon)

    # c. 准备数据并vmap
    data_for_xt = jnp.transpose(iq_data, (1, 0, 2))  # -> (z, x, t)
    data_for_xt_flat = data_for_xt.reshape(z_dim, -1)  # -> (z, x*t)

    # 使用vmap将函数应用到每个z切片上
    vmapped_interp_xt = jax.vmap(_rbf_interpolate_2d_jit, in_axes=(0, None, None, None, None, None))

    # 执行并行计算
    result_xt_flat = vmapped_interp_xt(
        data_for_xt_flat,
        known_indices_xt_flat,
        unknown_indices_xt_flat,
        phi_matrix_xt_inv,
        phi_new_matrix_xt,
        xt_shape,
    )
    f_rec_xt = jnp.transpose(result_xt_flat.reshape(z_dim, x_dim, t_dim), (1, 0, 2))  # 恢复 (x,z,t)
    print("x-t 平面处理完成。")

    # --- 3. 处理 z-t 平面 (并行处理所有x切片) ---
    print("开始并行处理 z-t 平面...")
    # a. 预计算 z-t 平面的几何信息
    zt_shape = (z_dim, t_dim)
    s_coords, t_coords = jnp.meshgrid(jnp.arange(z_dim), jnp.arange(t_dim), indexing="ij")
    all_coords_zt_flat = jnp.stack([s_coords.flatten(), t_coords.flatten()], axis=1)

    time_mask_2d_zt = jnp.tile(time_mask, z_dim)
    known_indices_zt_flat = jnp.where(time_mask_2d_zt)[0]
    unknown_indices_zt_flat = jnp.where(~time_mask_2d_zt)[0]

    known_coords_zt = all_coords_zt_flat[known_indices_zt_flat]
    unknown_coords_zt = all_coords_zt_flat[unknown_indices_zt_flat]

    # b. 预计算RBF矩阵
    dist_known_zt = jnp.linalg.norm(known_coords_zt[:, None, :] - known_coords_zt[None, :, :], axis=-1)
    phi_matrix_zt = multiquadric_kernel(dist_known_zt, epsilon)
    phi_matrix_zt_inv = jnp.linalg.inv(phi_matrix_zt)

    dist_new_zt = jnp.linalg.norm(unknown_coords_zt[:, None, :] - known_coords_zt[None, :, :], axis=-1)
    phi_new_matrix_zt = multiquadric_kernel(dist_new_zt, epsilon)

    # c. 准备数据并vmap
    data_for_zt_flat = iq_data.reshape(x_dim, -1)  # -> (x, z*t)
    vmapped_interp_zt = jax.vmap(_rbf_interpolate_2d_jit, in_axes=(0, None, None, None, None, None))
    result_zt_flat = vmapped_interp_zt(
        data_for_zt_flat,
        known_indices_zt_flat,
        unknown_indices_zt_flat,
        phi_matrix_zt_inv,
        phi_new_matrix_zt,
        zt_shape,
    )
    f_rec_zt = result_zt_flat.reshape(x_dim, z_dim, t_dim)
    print("z-t 平面处理完成。")

    # --- 4. 合并结果 ---
    print("合并结果...")
    final_reconstruction = (f_rec_xt + f_rec_zt) / 2.0
    final_reconstruction = jnp.where(jnp.isnan(iq_data), final_reconstruction, iq_data)

    return final_reconstruction


# --- 主程序：使用新的高性能函数 ---
if __name__ == "__main__":
    plt.rcParams["font.sans-serif"] = ["Source Han Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    # mpl.rcParams["axes.unicode_minus"] = False  # 设置正常显示符号
    # --- 1. 创建模拟数据 ---
    # 数据维度：(x_dim, z_dim, time_dim)
    x_dim, z_dim, t_dim = 200, 300, 50

    # 创建平滑变化的模拟信号
    x_range = jnp.linspace(-1, 1, x_dim)
    z_range = jnp.linspace(-1, 1, z_dim)
    t_range = jnp.linspace(0, 2 * jnp.pi, t_dim)

    # 使用meshgrid生成坐标
    xx, zz, tt = jnp.meshgrid(x_range, z_range, t_range, indexing="ij")
    center_x = 0.5 * jnp.sin(tt)
    center_z = 0.5 * jnp.cos(tt)
    ground_truth_data = jnp.exp(-((xx - center_x) ** 2 + (zz - center_z) ** 2) / 0.1)
    downsample_factor = 5
    sparse_data_np = np.array(ground_truth_data)
    for t_idx in range(t_dim):
        if t_idx % downsample_factor != 0:
            sparse_data_np[:, :, t_idx] = np.nan
    sparse_data = jnp.array(sparse_data_np)

    # --- 执行新的高性能插值函数 ---
    epsilon = 1e4
    print("--- 开始执行高性能并行版本 ---")
    reconstructed_data_v2 = rbf_2x2d_interpolate_3d_v2(sparse_data, epsilon)
    print("\n插值完成。")
    print(f"重建后是否还存在NaN值: {jnp.isnan(reconstructed_data_v2).any()}")

    # --- 可视化结果 (与之前相同) ---
    # ... 省略可视化代码 ...
    # 选择一个中间的x切片进行可视化
    slice_idx = t_dim // 2

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
