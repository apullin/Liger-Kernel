import operator

import torch
import triton
import triton.language as tl

from liger_kernel.ops.utils import compare_version
from liger_kernel.ops.utils import device_context
from liger_kernel.utils import infer_device
from liger_kernel.utils import is_npu_available

if compare_version("triton", operator.ge, "3.0.0") and not is_npu_available():
    try:
        from triton.language.extra.libdevice import rsqrt
    except ModuleNotFoundError:
        from triton.language.extra.cuda.libdevice import rsqrt
else:
    from triton.language.math import rsqrt

if infer_device() == "npu":
    MAX_FUSED_SIZE = 16384
else:
    MAX_FUSED_SIZE = 16384


def _is_channels_last_4d(t: torch.Tensor) -> bool:
    return (
        t.ndim == 4
        and t.size(1) > 1
        and (t.size(2) * t.size(3) > 1)
        and t.is_contiguous(memory_format=torch.channels_last)
    )


def _num_sms(device: torch.device) -> int:
    if device.type == "cuda":
        return torch.cuda.get_device_properties(device).multi_processor_count
    if device.type == "xpu":
        return torch.xpu.get_device_properties(device).gpu_eu_count
    return 32


# =============================================================================
# 1. Small-group / high-(N*G) single-CTA-per-group kernels (NCHW contiguous)
# =============================================================================


@triton.jit
def _gn_single_cta_fwd_kernel(
    Y_ptr,
    Y_group_stride,
    X_ptr,
    X_group_stride,
    Mean_ptr,
    RSTD_ptr,
    W_ptr,
    B_ptr,
    num_groups,
    channels_per_group,
    hw,
    eps,
    HAS_BIAS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    group_idx = pid % num_groups
    group_elems = channels_per_group * hw

    x_base = X_ptr + pid * X_group_stride
    y_base = Y_ptr + pid * Y_group_stride
    offs = tl.arange(0, BLOCK_SIZE)

    # Shifted one-pass variance in fp32 avoids E[x^2] - E[x]^2 cancellation at large offsets.
    shift = tl.load(x_base).to(tl.float32)
    sum_d = 0.0
    sum_d2 = 0.0
    for i in range(0, group_elems, BLOCK_SIZE):
        idx = i + offs
        mask = idx < group_elems
        x = tl.load(x_base + idx, mask=mask, other=0.0).to(tl.float32)
        d = tl.where(mask, x - shift, 0.0)
        sum_d += tl.sum(d, axis=0)
        sum_d2 += tl.sum(d * d, axis=0)

    mean_d = sum_d / group_elems
    mean = shift + mean_d
    var = tl.maximum(sum_d2 / group_elems - mean_d * mean_d, 0.0)
    rstd = rsqrt(var + tl.cast(eps, tl.float32))

    tl.store(Mean_ptr + pid, mean)
    tl.store(RSTD_ptr + pid, rstd)

    for c_local in range(0, channels_per_group):
        c = group_idx * channels_per_group + c_local
        w = tl.load(W_ptr + c).to(tl.float32)
        b = tl.load(B_ptr + c).to(tl.float32) if HAS_BIAS else 0.0
        ch_off = c_local * hw
        for i in range(0, hw, BLOCK_SIZE):
            idx = i + offs
            mask = idx < hw
            x = tl.load(x_base + ch_off + idx, mask=mask, other=0.0).to(tl.float32)
            y = (x - mean) * rstd * w + b
            tl.store(y_base + ch_off + idx, y, mask=mask)


@triton.jit
def _gn_single_cta_bwd_kernel(
    X_ptr,
    X_batch_stride,
    X_ch_stride,
    DY_ptr,
    DY_batch_stride,
    DY_ch_stride,
    DX_ptr,
    DX_batch_stride,
    DX_ch_stride,
    W_ptr,
    Mean_ptr,
    RSTD_ptr,
    DW_ptr,
    DB_ptr,
    num_groups,
    channels_per_group,
    hw,
    COMPUTE_DW: tl.constexpr,
    COMPUTE_DB: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    batch_idx = pid // num_groups
    group_idx = pid % num_groups

    mean = tl.load(Mean_ptr + pid).to(tl.float32)
    rstd = tl.load(RSTD_ptr + pid).to(tl.float32)

    x_b = X_ptr + batch_idx * X_batch_stride
    dy_b = DY_ptr + batch_idx * DY_batch_stride
    dx_b = DX_ptr + batch_idx * DX_batch_stride

    offs = tl.arange(0, BLOCK_SIZE)
    c1 = 0.0
    c2 = 0.0

    for c_local in range(0, channels_per_group):
        c = group_idx * channels_per_group + c_local
        w = tl.load(W_ptr + c).to(tl.float32)
        dw_acc = 0.0
        db_acc = 0.0
        for i in range(0, hw, BLOCK_SIZE):
            idx = i + offs
            mask = idx < hw
            x = tl.load(x_b + c * X_ch_stride + idx, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(dy_b + c * DY_ch_stride + idx, mask=mask, other=0.0).to(tl.float32)
            x_hat = tl.where(mask, (x - mean) * rstd, 0.0)
            dw_acc += tl.sum(dy * x_hat, axis=0)
            db_acc += tl.sum(dy, axis=0)
        c1 += w * dw_acc
        c2 += w * db_acc
        if COMPUTE_DW:
            tl.atomic_add(DW_ptr + c, dw_acc)
        if COMPUTE_DB:
            tl.atomic_add(DB_ptr + c, db_acc)

    inv_n = 1.0 / (channels_per_group * hw)
    c1 = c1 * inv_n
    c2 = c2 * inv_n

    for c_local in range(0, channels_per_group):
        c = group_idx * channels_per_group + c_local
        w = tl.load(W_ptr + c).to(tl.float32)
        for i in range(0, hw, BLOCK_SIZE):
            idx = i + offs
            mask = idx < hw
            x = tl.load(x_b + c * X_ch_stride + idx, mask=mask, other=0.0).to(tl.float32)
            dy = tl.load(dy_b + c * DY_ch_stride + idx, mask=mask, other=0.0).to(tl.float32)
            x_hat = (x - mean) * rstd
            dx = (w * dy - (x_hat * c1 + c2)) * rstd
            tl.store(dx_b + c * DX_ch_stride + idx, dx, mask=mask)


# =============================================================================
# 2. Split-Spatial NCHW Kernels (large HW / small N*G)
# =============================================================================


@triton.jit
def _gn_nchw_stats_kernel(
    X_ptr,
    PartSum_ptr,
    PartSum2_ptr,
    num_channels,
    num_groups,
    channels_per_group,
    hw,
    num_splits,
    chunk_hw,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    s = pid % num_splits
    ng = pid // num_splits
    n = ng // num_groups
    g = ng % num_groups

    hw_start = s * chunk_hw
    hw_end = tl.minimum(hw_start + chunk_hw, hw)

    # Group anchor x0 = X[n, g * channels_per_group, 0] for shift-invariant variance
    c0 = g * channels_per_group
    shift = tl.load(X_ptr + (n * num_channels + c0) * hw).to(tl.float32)

    offs = tl.arange(0, BLOCK_HW)
    sum_d = 0.0
    sum_d2 = 0.0

    if hw_start < hw_end:
        for c_local in range(0, channels_per_group):
            ch_base = X_ptr + (n * num_channels + c0 + c_local) * hw
            for i in range(hw_start, hw_end, BLOCK_HW):
                idx = i + offs
                mask = idx < hw_end
                x = tl.load(ch_base + idx, mask=mask, other=0.0).to(tl.float32)
                d = tl.where(mask, x - shift, 0.0)
                sum_d += tl.sum(d, axis=0)
                sum_d2 += tl.sum(d * d, axis=0)

    out_idx = ng * num_splits + s
    tl.store(PartSum_ptr + out_idx, sum_d)
    tl.store(PartSum2_ptr + out_idx, sum_d2)


@triton.jit
def _gn_nchw_apply_kernel(
    X_ptr,
    Y_ptr,
    Mean_ptr,
    RSTD_ptr,
    W_ptr,
    B_ptr,
    num_channels,
    channels_per_group,
    hw,
    tiles_per_channel,
    HAS_BIAS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    tile_id = pid % tiles_per_channel
    nc = pid // tiles_per_channel
    n = nc // num_channels
    c = nc % num_channels
    g = c // channels_per_group
    ng = n * (num_channels // channels_per_group) + g

    mean = tl.load(Mean_ptr + ng).to(tl.float32)
    rstd = tl.load(RSTD_ptr + ng).to(tl.float32)
    w = tl.load(W_ptr + c).to(tl.float32)
    b = tl.load(B_ptr + c).to(tl.float32) if HAS_BIAS else 0.0

    scale = rstd * w
    bias_eff = b - mean * scale

    idx = tile_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = idx < hw
    base = nc * hw + idx
    x = tl.load(X_ptr + base, mask=mask, other=0.0).to(tl.float32)
    y = x * scale + bias_eff
    tl.store(Y_ptr + base, y, mask=mask)


@triton.jit
def _gn_nchw_bwd_stats_kernel(
    X_ptr,
    DY_ptr,
    Mean_ptr,
    RSTD_ptr,
    U_ptr,
    V_ptr,
    num_channels,
    channels_per_group,
    hw,
    num_splits,
    chunk_hw,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    s = pid % num_splits
    nc = pid // num_splits
    n = nc // num_channels
    c = nc % num_channels
    g = c // channels_per_group
    ng = n * (num_channels // channels_per_group) + g

    mean = tl.load(Mean_ptr + ng).to(tl.float32)
    rstd = tl.load(RSTD_ptr + ng).to(tl.float32)

    hw_start = s * chunk_hw
    hw_end = tl.minimum(hw_start + chunk_hw, hw)
    offs = tl.arange(0, BLOCK_HW)

    u_acc = 0.0
    v_acc = 0.0
    ch_base = nc * hw
    for i in range(hw_start, hw_end, BLOCK_HW):
        idx = i + offs
        mask = idx < hw_end
        x = tl.load(X_ptr + ch_base + idx, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + ch_base + idx, mask=mask, other=0.0).to(tl.float32)
        x_hat = tl.where(mask, (x - mean) * rstd, 0.0)
        u_acc += tl.sum(dy * x_hat, axis=0)
        v_acc += tl.sum(dy, axis=0)

    tl.store(U_ptr + pid, u_acc)
    tl.store(V_ptr + pid, v_acc)


@triton.jit
def _gn_nchw_bwd_dx_kernel(
    X_ptr,
    DY_ptr,
    DX_ptr,
    Mean_ptr,
    RSTD_ptr,
    W_ptr,
    C1_ptr,
    C2_ptr,
    num_channels,
    channels_per_group,
    hw,
    tiles_per_channel,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    tile_id = pid % tiles_per_channel
    nc = pid // tiles_per_channel
    n = nc // num_channels
    c = nc % num_channels
    g = c // channels_per_group
    ng = n * (num_channels // channels_per_group) + g

    mean = tl.load(Mean_ptr + ng).to(tl.float32)
    rstd = tl.load(RSTD_ptr + ng).to(tl.float32)
    w = tl.load(W_ptr + c).to(tl.float32)
    c1 = tl.load(C1_ptr + ng).to(tl.float32)
    c2 = tl.load(C2_ptr + ng).to(tl.float32)

    idx = tile_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask = idx < hw
    base = nc * hw + idx
    x = tl.load(X_ptr + base, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY_ptr + base, mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean) * rstd
    dx = (w * dy - (x_hat * c1 + c2)) * rstd
    tl.store(DX_ptr + base, dx, mask=mask)


# =============================================================================
# 3. Native channels_last (NHWC: physical [N, HW, C]) Kernels (zero transpose!)
# =============================================================================


@triton.jit
def _gn_nhwc_stats_kernel(
    X_ptr,
    Shift_ptr,
    PartSum_ptr,
    PartSum2_ptr,
    num_channels,
    channels_per_group,
    hw,
    num_splits,
    chunk_hw,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    s = pid % num_splits
    n = pid // num_splits

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < num_channels
    g_offs = c_offs // channels_per_group

    # Shift per group g: X[n, 0, g * channels_per_group]
    shift_c = tl.load(Shift_ptr + n * (num_channels // channels_per_group) + g_offs, mask=c_mask, other=0.0).to(
        tl.float32
    )

    hw_start = s * chunk_hw
    hw_end = tl.minimum(hw_start + chunk_hw, hw)

    sum_d = tl.zeros((BLOCK_C,), dtype=tl.float32)
    sum_d2 = tl.zeros((BLOCK_C,), dtype=tl.float32)
    x_n_base = X_ptr + n * hw * num_channels

    for i in range(hw_start, hw_end, BLOCK_HW):
        r_offs = i + tl.arange(0, BLOCK_HW)
        r_mask = r_offs < hw_end
        mask = r_mask[:, None] & c_mask[None, :]
        x = tl.load(x_n_base + r_offs[:, None] * num_channels + c_offs[None, :], mask=mask, other=0.0).to(tl.float32)
        d = tl.where(mask, x - shift_c[None, :], 0.0)
        sum_d += tl.sum(d, axis=0)
        sum_d2 += tl.sum(d * d, axis=0)

    out_base = pid * num_channels + c_offs
    tl.store(PartSum_ptr + out_base, sum_d, mask=c_mask)
    tl.store(PartSum2_ptr + out_base, sum_d2, mask=c_mask)


@triton.jit
def _gn_nhwc_apply_kernel(
    X_ptr,
    Y_ptr,
    Mean_ptr,
    RSTD_ptr,
    W_ptr,
    B_ptr,
    num_channels,
    channels_per_group,
    hw,
    tiles_per_n,
    HAS_BIAS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    tile_id = pid % tiles_per_n
    n = pid // tiles_per_n
    num_groups = num_channels // channels_per_group

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < num_channels
    g_offs = c_offs // channels_per_group

    mean_c = tl.load(Mean_ptr + n * num_groups + g_offs, mask=c_mask, other=0.0).to(tl.float32)
    rstd_c = tl.load(RSTD_ptr + n * num_groups + g_offs, mask=c_mask, other=0.0).to(tl.float32)
    w_c = tl.load(W_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    b_c = (
        tl.load(B_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
        if HAS_BIAS
        else tl.zeros((BLOCK_C,), dtype=tl.float32)
    )

    scale_c = rstd_c * w_c
    bias_c = b_c - mean_c * scale_c

    r_offs = tile_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
    r_mask = r_offs < hw
    mask = r_mask[:, None] & c_mask[None, :]
    offsets = (n * hw + r_offs[:, None]) * num_channels + c_offs[None, :]

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x * scale_c[None, :] + bias_c[None, :]
    tl.store(Y_ptr + offsets, y, mask=mask)


@triton.jit
def _gn_nhwc_bwd_stats_kernel(
    X_ptr,
    DY_ptr,
    Mean_ptr,
    RSTD_ptr,
    U_ptr,
    V_ptr,
    num_channels,
    channels_per_group,
    hw,
    num_splits,
    chunk_hw,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    s = pid % num_splits
    n = pid // num_splits
    num_groups = num_channels // channels_per_group

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < num_channels
    g_offs = c_offs // channels_per_group

    mean_c = tl.load(Mean_ptr + n * num_groups + g_offs, mask=c_mask, other=0.0).to(tl.float32)
    rstd_c = tl.load(RSTD_ptr + n * num_groups + g_offs, mask=c_mask, other=0.0).to(tl.float32)

    hw_start = s * chunk_hw
    hw_end = tl.minimum(hw_start + chunk_hw, hw)

    u_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    v_acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    base_n = n * hw * num_channels

    for i in range(hw_start, hw_end, BLOCK_HW):
        r_offs = i + tl.arange(0, BLOCK_HW)
        r_mask = r_offs < hw_end
        mask = r_mask[:, None] & c_mask[None, :]
        offs = base_n + r_offs[:, None] * num_channels + c_offs[None, :]
        x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        x_hat = tl.where(mask, (x - mean_c[None, :]) * rstd_c[None, :], 0.0)
        u_acc += tl.sum(dy * x_hat, axis=0)
        v_acc += tl.sum(tl.where(mask, dy, 0.0), axis=0)

    out_base = pid * num_channels + c_offs
    tl.store(U_ptr + out_base, u_acc, mask=c_mask)
    tl.store(V_ptr + out_base, v_acc, mask=c_mask)


@triton.jit
def _gn_nhwc_bwd_dx_kernel(
    X_ptr,
    DY_ptr,
    DX_ptr,
    Mean_ptr,
    RSTD_ptr,
    W_ptr,
    C1_ptr,
    C2_ptr,
    num_channels,
    channels_per_group,
    hw,
    tiles_per_n,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)
    tile_id = pid % tiles_per_n
    n = pid // tiles_per_n
    num_groups = num_channels // channels_per_group

    c_offs = tl.arange(0, BLOCK_C)
    c_mask = c_offs < num_channels
    g_offs = c_offs // channels_per_group

    mean_c = tl.load(Mean_ptr + n * num_groups + g_offs, mask=c_mask, other=0.0).to(tl.float32)
    rstd_c = tl.load(RSTD_ptr + n * num_groups + g_offs, mask=c_mask, other=0.0).to(tl.float32)
    w_c = tl.load(W_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    c1_c = tl.load(C1_ptr + n * num_groups + g_offs, mask=c_mask, other=0.0).to(tl.float32)
    c2_c = tl.load(C2_ptr + n * num_groups + g_offs, mask=c_mask, other=0.0).to(tl.float32)

    r_offs = tile_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
    r_mask = r_offs < hw
    mask = r_mask[:, None] & c_mask[None, :]
    offsets = (n * hw + r_offs[:, None]) * num_channels + c_offs[None, :]

    x = tl.load(X_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    dy = tl.load(DY_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    x_hat = (x - mean_c[None, :]) * rstd_c[None, :]
    dx = (w_c[None, :] * dy - (x_hat * c1_c[None, :] + c2_c[None, :])) * rstd_c[None, :]
    tl.store(DX_ptr + offsets, dx, mask=mask)


# =============================================================================
# Host Dispatchers & Autograd Function
# =============================================================================


def group_norm_forward(X, num_channels, num_groups, W, B, eps):
    shape = X.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    has_bias = B is not None
    W = W.contiguous()
    if has_bias:
        B = B.contiguous()

    # Fast path 1: 4D channels_last (NHWC) with C <= 256 — zero transpose copies!
    if _is_channels_last_4d(X) and num_channels <= 256:
        hw = shape[2] * shape[3]
        X_nhwc = X.permute(0, 2, 3, 1).reshape(batch_size, hw, num_channels)
        Y = torch.empty_like(X, memory_format=torch.channels_last)
        Y_nhwc = Y.permute(0, 2, 3, 1).reshape(batch_size, hw, num_channels)

        BLOCK_C = max(16, triton.next_power_of_2(num_channels))
        BLOCK_HW = min(64, max(16, 4096 // BLOCK_C))
        sms = _num_sms(X.device)
        num_splits = max(1, min(triton.cdiv(hw, BLOCK_HW), triton.cdiv(sms * 4, max(1, batch_size))))
        chunk_hw = triton.cdiv(hw, num_splits)

        shift = X_nhwc[:, 0, ::channels_per_group].contiguous().to(torch.float32)  # [N, G]
        part_sum = torch.empty((batch_size, num_splits, num_channels), dtype=torch.float32, device=X.device)
        part_sum2 = torch.empty((batch_size, num_splits, num_channels), dtype=torch.float32, device=X.device)

        with device_context(X.device):
            _gn_nhwc_stats_kernel[(batch_size * num_splits,)](
                X_nhwc,
                shift,
                part_sum,
                part_sum2,
                num_channels,
                channels_per_group,
                hw,
                num_splits,
                chunk_hw,
                BLOCK_HW=BLOCK_HW,
                BLOCK_C=BLOCK_C,
                num_warps=4,
            )

            group_elems = float(channels_per_group * hw)
            sum_d = part_sum.sum(dim=1).view(batch_size, num_groups, channels_per_group).sum(dim=-1)
            sum_d2 = part_sum2.sum(dim=1).view(batch_size, num_groups, channels_per_group).sum(dim=-1)
            mean_d = sum_d / group_elems
            Mean = shift + mean_d
            var = torch.clamp(sum_d2 / group_elems - mean_d * mean_d, min=0.0)
            RSTD = torch.rsqrt(var + eps)

            tiles_per_n = triton.cdiv(hw, BLOCK_HW)
            _gn_nhwc_apply_kernel[(batch_size * tiles_per_n,)](
                X_nhwc,
                Y_nhwc,
                Mean,
                RSTD,
                W,
                B,
                num_channels,
                channels_per_group,
                hw,
                tiles_per_n,
                HAS_BIAS=has_bias,
                BLOCK_HW=BLOCK_HW,
                BLOCK_C=BLOCK_C,
                num_warps=4,
            )
        return Y, X, Mean, RSTD, BLOCK_HW

    # Contiguous NCHW / (N, C, *) path
    X = X.contiguous()
    hw = X.numel() // (batch_size * num_channels)
    group_elems = channels_per_group * hw
    sms = _num_sms(X.device)

    # Fast path 2: Split-spatial NCHW when (N * G) under-utilizes SMs or group is large
    if hw >= 4096 and (batch_size * num_groups < 2 * sms or group_elems > MAX_FUSED_SIZE):
        X_flat = X.view(batch_size * num_channels, hw)
        Y = torch.empty_like(X)
        Y_flat = Y.view(batch_size * num_channels, hw)

        BLOCK_HW = 1024 if hw < 8192 else 2048
        target_blocks = sms * 4
        num_splits = max(
            1, min(128, triton.cdiv(hw, BLOCK_HW), triton.cdiv(target_blocks, max(1, batch_size * num_groups)))
        )
        chunk_hw = triton.cdiv(hw, num_splits)

        part_sum = torch.empty((batch_size * num_groups, num_splits), dtype=torch.float32, device=X.device)
        part_sum2 = torch.empty((batch_size * num_groups, num_splits), dtype=torch.float32, device=X.device)

        with device_context(X.device):
            _gn_nchw_stats_kernel[(batch_size * num_groups * num_splits,)](
                X_flat,
                part_sum,
                part_sum2,
                num_channels,
                num_groups,
                channels_per_group,
                hw,
                num_splits,
                chunk_hw,
                BLOCK_HW=BLOCK_HW,
                num_warps=4 if BLOCK_HW <= 1024 else 8,
            )

            shift = X.view(batch_size, num_groups, group_elems)[:, :, 0].to(torch.float32)
            mean_d = part_sum.sum(dim=-1).view(batch_size, num_groups) / float(group_elems)
            sum_d2 = part_sum2.sum(dim=-1).view(batch_size, num_groups)
            Mean = (shift + mean_d).contiguous()
            var = torch.clamp(sum_d2 / float(group_elems) - mean_d * mean_d, min=0.0)
            RSTD = torch.rsqrt(var + eps).contiguous()

            tiles_per_channel = triton.cdiv(hw, BLOCK_HW)
            _gn_nchw_apply_kernel[(batch_size * num_channels * tiles_per_channel,)](
                X_flat,
                Y_flat,
                Mean,
                RSTD,
                W,
                B,
                num_channels,
                channels_per_group,
                hw,
                tiles_per_channel,
                HAS_BIAS=has_bias,
                BLOCK_HW=BLOCK_HW,
                num_warps=4 if BLOCK_HW <= 1024 else 8,
            )
        return Y, X, Mean, RSTD, BLOCK_HW

    # Path 3: Small-spatial / high-(N*G) single-CTA-per-group path
    X_grp = X.view(batch_size, num_groups, group_elems)
    Y_grp = torch.empty_like(X_grp)
    Mean = torch.empty((batch_size, num_groups), dtype=torch.float32, device=X.device)
    RSTD = torch.empty((batch_size, num_groups), dtype=torch.float32, device=X.device)

    BLOCK_SIZE = min(MAX_FUSED_SIZE, max(16, triton.next_power_of_2(max(hw, group_elems))))
    num_warps = 8 if BLOCK_SIZE >= 2048 else 4

    with device_context(X.device):
        _gn_single_cta_fwd_kernel[(batch_size * num_groups,)](
            Y_grp,
            Y_grp.stride(1),
            X_grp,
            X_grp.stride(1),
            Mean,
            RSTD,
            W,
            B,
            num_groups,
            channels_per_group,
            hw,
            eps,
            HAS_BIAS=has_bias,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    return Y_grp.view(*shape), X, Mean, RSTD, BLOCK_SIZE


def group_norm_backward(dY, X, W, B, Mean, RSTD, num_channels, num_groups, needs_dw=True, needs_db=True):
    shape = X.shape
    batch_size = shape[0]
    channels_per_group = num_channels // num_groups
    compute_dw = bool(needs_dw)
    compute_db = (B is not None) and bool(needs_db)
    W = W.contiguous()

    # Fast path 1: 4D channels_last (NHWC)
    if _is_channels_last_4d(X) and num_channels <= 256:
        hw = shape[2] * shape[3]
        if not _is_channels_last_4d(dY):
            dY = dY.contiguous(memory_format=torch.channels_last)
        X_nhwc = X.permute(0, 2, 3, 1).reshape(batch_size, hw, num_channels)
        dY_nhwc = dY.permute(0, 2, 3, 1).reshape(batch_size, hw, num_channels)
        DX = torch.empty_like(X, memory_format=torch.channels_last)
        DX_nhwc = DX.permute(0, 2, 3, 1).reshape(batch_size, hw, num_channels)

        BLOCK_C = max(16, triton.next_power_of_2(num_channels))
        BLOCK_HW = min(64, max(16, 4096 // BLOCK_C))
        sms = _num_sms(X.device)
        num_splits = max(1, min(triton.cdiv(hw, BLOCK_HW), triton.cdiv(sms * 4, max(1, batch_size))))
        chunk_hw = triton.cdiv(hw, num_splits)

        U = torch.empty((batch_size * num_splits, num_channels), dtype=torch.float32, device=X.device)
        V = torch.empty((batch_size * num_splits, num_channels), dtype=torch.float32, device=X.device)

        with device_context(X.device):
            _gn_nhwc_bwd_stats_kernel[(batch_size * num_splits,)](
                X_nhwc,
                dY_nhwc,
                Mean,
                RSTD,
                U,
                V,
                num_channels,
                channels_per_group,
                hw,
                num_splits,
                chunk_hw,
                BLOCK_HW=BLOCK_HW,
                BLOCK_C=BLOCK_C,
                num_warps=4,
            )

            u_nc = U.view(batch_size, num_splits, num_channels).sum(dim=1)  # [N, C]
            v_nc = V.view(batch_size, num_splits, num_channels).sum(dim=1)  # [N, C]
            DW = u_nc.sum(dim=0).to(W.dtype) if compute_dw else None
            DB = v_nc.sum(dim=0).to(B.dtype) if compute_db else None

            w_f32 = W.to(torch.float32)
            inv_n = 1.0 / float(channels_per_group * hw)
            C1 = ((u_nc * w_f32).view(batch_size, num_groups, channels_per_group).sum(dim=-1) * inv_n).contiguous()
            C2 = ((v_nc * w_f32).view(batch_size, num_groups, channels_per_group).sum(dim=-1) * inv_n).contiguous()

            tiles_per_n = triton.cdiv(hw, BLOCK_HW)
            _gn_nhwc_bwd_dx_kernel[(batch_size * tiles_per_n,)](
                X_nhwc,
                dY_nhwc,
                DX_nhwc,
                Mean,
                RSTD,
                W,
                C1,
                C2,
                num_channels,
                channels_per_group,
                hw,
                tiles_per_n,
                BLOCK_HW=BLOCK_HW,
                BLOCK_C=BLOCK_C,
                num_warps=4,
            )
        return DX, DW, DB

    # Contiguous NCHW / (N, C, *) path
    X = X.contiguous()
    dY = dY.contiguous()
    hw = X.numel() // (batch_size * num_channels)
    group_elems = channels_per_group * hw
    sms = _num_sms(X.device)

    # Fast path 2: Split-spatial NCHW
    if hw >= 4096 and (batch_size * num_groups < 2 * sms or group_elems > MAX_FUSED_SIZE):
        X_flat = X.view(batch_size * num_channels, hw)
        dY_flat = dY.view(batch_size * num_channels, hw)
        DX = torch.empty_like(X)
        DX_flat = DX.view(batch_size * num_channels, hw)

        BLOCK_HW = 1024 if hw < 8192 else 2048
        target_blocks = sms * 4
        num_splits = max(
            1, min(128, triton.cdiv(hw, BLOCK_HW), triton.cdiv(target_blocks, max(1, batch_size * num_channels)))
        )
        chunk_hw = triton.cdiv(hw, num_splits)

        U = torch.empty((batch_size * num_channels, num_splits), dtype=torch.float32, device=X.device)
        V = torch.empty((batch_size * num_channels, num_splits), dtype=torch.float32, device=X.device)

        with device_context(X.device):
            _gn_nchw_bwd_stats_kernel[(batch_size * num_channels * num_splits,)](
                X_flat,
                dY_flat,
                Mean,
                RSTD,
                U,
                V,
                num_channels,
                channels_per_group,
                hw,
                num_splits,
                chunk_hw,
                BLOCK_HW=BLOCK_HW,
                num_warps=4 if BLOCK_HW <= 1024 else 8,
            )

            u_nc = U.sum(dim=-1).view(batch_size, num_channels)
            v_nc = V.sum(dim=-1).view(batch_size, num_channels)
            DW = u_nc.sum(dim=0).to(W.dtype) if compute_dw else None
            DB = v_nc.sum(dim=0).to(B.dtype) if compute_db else None

            w_f32 = W.to(torch.float32)
            inv_n = 1.0 / float(group_elems)
            C1 = ((u_nc * w_f32).view(batch_size, num_groups, channels_per_group).sum(dim=-1) * inv_n).contiguous()
            C2 = ((v_nc * w_f32).view(batch_size, num_groups, channels_per_group).sum(dim=-1) * inv_n).contiguous()

            tiles_per_channel = triton.cdiv(hw, BLOCK_HW)
            _gn_nchw_bwd_dx_kernel[(batch_size * num_channels * tiles_per_channel,)](
                X_flat,
                dY_flat,
                DX_flat,
                Mean,
                RSTD,
                W,
                C1,
                C2,
                num_channels,
                channels_per_group,
                hw,
                tiles_per_channel,
                BLOCK_HW=BLOCK_HW,
                num_warps=4 if BLOCK_HW <= 1024 else 8,
            )
        return DX, DW, DB

    # Path 3: Small-spatial / high-(N*G) single-CTA-per-group path
    X_3d = X.view(batch_size, num_channels, hw)
    dY_3d = dY.view(batch_size, num_channels, hw)
    DX_3d = torch.empty_like(X_3d)
    DW_f32 = torch.zeros((num_channels,), dtype=torch.float32, device=W.device) if compute_dw else None
    DB_f32 = torch.zeros((num_channels,), dtype=torch.float32, device=B.device) if compute_db else None

    BLOCK_SIZE = min(MAX_FUSED_SIZE, max(16, triton.next_power_of_2(hw)))
    num_warps = 8 if BLOCK_SIZE >= 2048 else 4

    with device_context(X.device):
        _gn_single_cta_bwd_kernel[(batch_size * num_groups,)](
            X_3d,
            X_3d.stride(0),
            X_3d.stride(1),
            dY_3d,
            dY_3d.stride(0),
            dY_3d.stride(1),
            DX_3d,
            DX_3d.stride(0),
            DX_3d.stride(1),
            W,
            Mean,
            RSTD,
            DW_f32,
            DB_f32,
            num_groups,
            channels_per_group,
            hw,
            COMPUTE_DW=compute_dw,
            COMPUTE_DB=compute_db,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
    DW = DW_f32.to(W.dtype) if compute_dw else None
    DB = DB_f32.to(B.dtype) if compute_db else None
    return DX_3d.view(*shape), DW, DB


class LigerGroupNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        X,
        affine_scaling_weight,
        affine_shifting_bias,
        num_channels,
        num_groups,
        eps,
    ):
        Y, X_saved, Mean, RSTD, _ = group_norm_forward(
            X,
            num_channels,
            num_groups,
            affine_scaling_weight,
            affine_shifting_bias,
            eps,
        )
        ctx.num_channels = num_channels
        ctx.num_groups = num_groups
        ctx.save_for_backward(X_saved, affine_scaling_weight, affine_shifting_bias, Mean, RSTD)
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, W, B, Mean, RSTD = ctx.saved_tensors
        needs_dw = ctx.needs_input_grad[1]
        needs_db = B is not None and ctx.needs_input_grad[2]
        DX, DW, DB = group_norm_backward(
            dY, X, W, B, Mean, RSTD, ctx.num_channels, ctx.num_groups, needs_dw=needs_dw, needs_db=needs_db
        )
        return DX, DW, DB, None, None, None
