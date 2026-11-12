import torch
import triton
import triton.language as tl


def _token_row(x):
    if x.size(1) == 1 and x.is_contiguous():
        return x, x.stride(0)
    x = x.contiguous()
    return x, x.stride(1)


def _prepare_freqs_cis(freqs_cis, batch_size: int, seq_len: int):
    if freqs_cis.is_complex():
        if freqs_cis.ndim == 4 and freqs_cis.shape[2] == 1:
            freqs_cis = freqs_cis.squeeze(2)
        if freqs_cis.ndim == 2:
            freqs_cis = freqs_cis[:seq_len].unsqueeze(0)
        elif freqs_cis.ndim == 3:
            if freqs_cis.shape[0] not in (1, batch_size):
                raise ValueError(
                    f"freqs_cis batch dimension must be 1 or batch_size ({batch_size}), got {freqs_cis.shape[0]}"
                )
            freqs_cis = freqs_cis[:, :seq_len]
        else:
            raise ValueError(f"Unsupported complex freqs_cis shape: {tuple(freqs_cis.shape)}")
        freqs_real = torch.view_as_real(freqs_cis)
    else:
        if freqs_cis.ndim == 3 and freqs_cis.shape[-1] == 2:
            freqs_real = freqs_cis[:seq_len].unsqueeze(0)
        elif freqs_cis.ndim == 4 and freqs_cis.shape[-1] == 2:
            if freqs_cis.shape[0] not in (1, batch_size):
                raise ValueError(
                    f"freqs_cis batch dimension must be 1 or batch_size ({batch_size}), got {freqs_cis.shape[0]}"
                )
            freqs_real = freqs_cis[:, :seq_len]
        else:
            raise ValueError(f"Unsupported real freqs_cis shape: {tuple(freqs_cis.shape)}")

    freqs_real = freqs_real.contiguous()
    freqs_batch_stride = 0 if freqs_real.shape[0] == 1 else freqs_real.stride(0)
    freqs_row_stride = freqs_real.stride(1)
    return freqs_real, freqs_batch_stride, freqs_row_stride


def _cast_and_contiguous(q, k):
    # Align dtype: fp32 only when q is fp32; otherwise keep q dtype for perf
    compute_dtype = torch.float32 if q.dtype == torch.float32 else q.dtype

    if k.dtype != q.dtype:
        k = k.to(q.dtype)

    q, q_row_stride = _token_row(q.to(compute_dtype))
    k, k_row_stride = _token_row(k.to(compute_dtype))
    return q, q_row_stride, k, k_row_stride


@triton.jit
def _llama4_rope_kernel(
    q_ptr,
    k_ptr,
    freqs_complex_ptr,
    q_row_stride,
    k_row_stride,
    q_head_stride,
    k_head_stride,
    freqs_batch_stride,
    freqs_row_stride,
    seq_len,
    batch_size,
    imag_sign,
    head_dim_half: tl.constexpr,
    n_q_heads: tl.constexpr,
    n_k_heads: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    H100-optimized RoPE kernel with improved parallelization across heads and dimensions.
    Grid: (batch*seq, head)
    """
    # 2D grid
    pid_bs = tl.program_id(0).to(tl.int64)  # over batch*seq
    pid_h = tl.program_id(1)  # over heads

    batch_idx = pid_bs // seq_len
    seq_idx = pid_bs % seq_len

    # Bounds check
    if batch_idx >= batch_size or seq_idx >= seq_len:
        return

    # Base pointers for this (batch, seq) position
    base_offset = batch_idx * seq_len + seq_idx
    q_base = q_ptr + base_offset * q_row_stride
    k_base = k_ptr + base_offset * k_row_stride
    freq_base = batch_idx * freqs_batch_stride + seq_idx * freqs_row_stride

    # Tiling over dim/2
    for d_start in tl.static_range(0, head_dim_half, BLOCK_SIZE):
        d_indices = d_start + tl.arange(0, BLOCK_SIZE)
        mask_d = d_indices < head_dim_half

        # Compute offsets for the block
        freq_offsets = d_indices[:, None] * 2 + tl.arange(0, 2)[None, :]
        # Load the block
        freqs_complex = tl.load(freqs_complex_ptr + freq_base + freq_offsets, mask=mask_d[:, None], other=0.0)
        freqs_real, freqs_imag = tl.split(freqs_complex)
        freqs_imag = freqs_imag * imag_sign

        # Process one query head per program in pid_h
        if pid_h < n_q_heads:
            q_head_ptr = q_base + pid_h * q_head_stride
            q_real = tl.load(q_head_ptr + d_indices * 2, mask=mask_d, other=0.0)
            q_imag = tl.load(q_head_ptr + d_indices * 2 + 1, mask=mask_d, other=0.0)

            # Complex multiply with FMAs: (a+ib)*(c+i d) = (a*c - b*d) + i(a*d + b*c)
            new_q_real = tl.math.fma(q_real, freqs_real, -(q_imag * freqs_imag))
            new_q_imag = tl.math.fma(q_real, freqs_imag, q_imag * freqs_real)

            tl.store(q_head_ptr + d_indices * 2, new_q_real, mask=mask_d)
            tl.store(q_head_ptr + d_indices * 2 + 1, new_q_imag, mask=mask_d)

        # Process one key head per program in pid_h
        if pid_h < n_k_heads:
            k_head_ptr = k_base + pid_h * k_head_stride
            k_real = tl.load(k_head_ptr + d_indices * 2, mask=mask_d, other=0.0)
            k_imag = tl.load(k_head_ptr + d_indices * 2 + 1, mask=mask_d, other=0.0)

            new_k_real = tl.math.fma(k_real, freqs_real, -(k_imag * freqs_imag))
            new_k_imag = tl.math.fma(k_real, freqs_imag, k_imag * freqs_real)

            tl.store(k_head_ptr + d_indices * 2, new_k_real, mask=mask_d)
            tl.store(k_head_ptr + d_indices * 2 + 1, new_k_imag, mask=mask_d)


def _select_kernel_meta(head_dim_half: int):
    # Heuristic tuning for block size and num_warps
    if head_dim_half >= 256:
        return 128, 8
    if head_dim_half >= 96:
        return 128, 4
    if head_dim_half >= 48:
        return 64, 4
    if head_dim_half >= 24:
        return 32, 2
    return 16, 2


def llama4_rope_forward(q, k, freqs_cis, BLOCK_SIZE: int = None, imag_sign: float = 1.0):
    # Save original dtypes for casting back
    original_q_dtype = q.dtype
    original_k_dtype = k.dtype

    batch_size, seq_len, n_q_heads, head_dim = q.shape
    _, _, n_k_heads, _ = k.shape
    head_dim_half = head_dim // 2

    freqs_real, freqs_batch_stride, freqs_row_stride = _prepare_freqs_cis(freqs_cis, batch_size, seq_len)

    # Cast to appropriate dtype and resolve token row strides
    q, q_row_stride, k, k_row_stride = _cast_and_contiguous(q, k)

    # H100-optimized meta-params
    if BLOCK_SIZE is None:
        BLOCK_SIZE, num_warps = _select_kernel_meta(head_dim_half)
    else:
        # Provide a default num_warps if caller pins BLOCK_SIZE
        _, num_warps = _select_kernel_meta(head_dim_half)

    # 2D grid: one program per (batch, seq, head)
    n_heads_max = max(n_q_heads, n_k_heads)
    grid = (batch_size * seq_len, n_heads_max)

    # Launch kernel
    _llama4_rope_kernel[grid](
        q,
        k,
        freqs_real,
        q_row_stride,
        k_row_stride,
        q.stride(2),
        k.stride(2),
        freqs_batch_stride,
        freqs_row_stride,
        seq_len,
        batch_size,
        imag_sign,
        head_dim_half,
        n_q_heads,
        n_k_heads,
        BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=2,
    )

    # Cast back to original dtypes only if they differ from compute dtype
    if q.dtype != original_q_dtype:
        q = q.to(original_q_dtype)
    if k.dtype != original_k_dtype:
        k = k.to(original_k_dtype)

    return q, k


class LigerLlama4RopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, freqs_cis, BLOCK_SIZE: int = None):
        q_out, k_out = llama4_rope_forward(q, k, freqs_cis, BLOCK_SIZE, imag_sign=1.0)
        ctx.save_for_backward(freqs_cis.detach() if isinstance(freqs_cis, torch.Tensor) else freqs_cis)
        ctx.BLOCK_SIZE = BLOCK_SIZE
        return q_out, k_out

    @staticmethod
    def backward(ctx, dq, dk):
        (freqs_cis,) = ctx.saved_tensors
        BLOCK_SIZE = getattr(ctx, "BLOCK_SIZE", None)
        # Use imag_sign=-1.0 for conjugate without materializing a new tensor
        dq_out, dk_out = llama4_rope_forward(dq, dk, freqs_cis, BLOCK_SIZE, imag_sign=-1.0)
        return dq_out, dk_out, None
