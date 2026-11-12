import os

import pytest
import torch
import torch.nn as nn

from test.utils import assert_verbose_allclose
from test.utils import set_seed
from test.utils import supports_bfloat16

from liger_kernel.ops import LigerPolyNormFunction
from liger_kernel.transformers.functional import liger_poly_norm
from liger_kernel.transformers.poly_norm import LigerPolyNorm
from liger_kernel.utils import infer_device

device = infer_device()

set_seed(42)
torch.use_deterministic_algorithms(True)

#  Only setting torch.use_deterministic_algorithms(True) might throw the following error:
#  RuntimeError: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`,
#  but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an
#  environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information,
#  go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility

if device == "cuda":
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

SLEEP_SECONDS = 0.1


class NaivePolyNorm(nn.Module):
    """
    Naive PyTorch implementation of PolyNorm for testing.

    Reference implementation from:
    https://github.com/BryceZhuo/PolyCom/

    PolyNorm formula:
        y = w₀·norm(x³) + w₁·norm(x²) + w₂·norm(x) + b
        where norm(u) = u / sqrt(mean(u²) + ε)
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.full((3,), 1.0 / 3.0))
        self.bias = nn.Parameter(torch.tensor(1.0))
        self.eps = eps

    def _norm(self, x):
        """RMSNorm operation"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass of PolyNorm

        Args:
            x: input tensor of shape (..., H)

        Returns:
            output tensor of same shape as input
        """
        # Compute powers
        x_pow3 = x**3
        x_pow2 = x**2
        x_pow1 = x**1

        # Normalize each power
        norm_x3 = self._norm(x_pow3)
        norm_x2 = self._norm(x_pow2)
        norm_x1 = self._norm(x_pow1)

        # Weighted sum with bias
        output = self.weight[0] * norm_x3 + self.weight[1] * norm_x2 + self.weight[2] * norm_x1 + self.bias

        return output


@pytest.mark.flaky(reruns=3, reruns_delay=2)
@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 128, 512),
        (8, 64, 1024),
        # weird shapes
        (5, 123, 123),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        # atol/rtol are slightly looser than test_correctness_functional's because weight.grad and
        # bias.grad here are sum() reductions over bs*sl*hd (up to ~524k) near-canceling terms. Triton's
        # and PyTorch's reductions accumulate in different (non-associative) orders, so their results can
        # differ from each other by a bit more than machine-precision-tight tolerances, even though both
        # are individually close to the true (higher-precision) gradient.
        (torch.float32, 3e-4, 1e-6),
        pytest.param(
            torch.bfloat16,
            6e-1,
            5e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness(bs, sl, hd, dtype, atol, rtol):
    """
    Test LigerPolyNorm wrapper correctness against naive PyTorch implementation.

    Args:
        bs: batch size
        sl: sequence length
        hd: hidden dimension
        dtype: data type (float32 or bfloat16)
        atol: absolute tolerance
        rtol: relative tolerance
    """
    _tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    x1 = _tensor.clone().requires_grad_(True)
    x2 = _tensor.clone().requires_grad_(True)

    # Gradient output
    grad_output = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    # Reference: Naive PyTorch implementation
    naive_poly_norm = NaivePolyNorm(eps=1e-6).to(device).to(dtype)
    ref_output = naive_poly_norm(x1)
    ref_output.backward(grad_output, retain_graph=True)

    # Liger wrapper implementation
    liger_poly_norm = LigerPolyNorm(eps=1e-6).to(device).to(dtype)
    # Copy weights to ensure same initialization
    liger_poly_norm.weight.data.copy_(naive_poly_norm.weight.data)
    liger_poly_norm.bias.data.copy_(naive_poly_norm.bias.data)

    triton_output = liger_poly_norm(x2)
    triton_output.backward(grad_output, retain_graph=True)

    # Check forward pass
    assert_verbose_allclose(ref_output, triton_output, atol=atol, rtol=rtol)

    # Check weight gradient
    assert_verbose_allclose(naive_poly_norm.weight.grad, liger_poly_norm.weight.grad, atol=atol, rtol=rtol)

    # Check bias gradient
    assert_verbose_allclose(naive_poly_norm.bias.grad, liger_poly_norm.bias.grad, atol=atol, rtol=rtol)

    # Check input gradient
    assert_verbose_allclose(x1.grad, x2.grad, atol=atol, rtol=rtol, max_print=20)


@pytest.mark.parametrize(
    "shape",
    [
        (8, 4096),
        (8, 8192),
        (2048, 8192),  # MLP activation shape of a 2.6B model using PolyNorm
    ],
)
def test_correctness_fp16_large_hidden(shape):
    """
    For unit-variance fp16 input, sum(x^6) exceeds the fp16 max (65504) at hidden size ~4096,
    so the forward must accumulate in fp32. Compare against the naive implementation in fp32.
    """
    torch.manual_seed(0)
    x = torch.randn(*shape, device=device, dtype=torch.float16)

    naive_poly_norm = NaivePolyNorm(eps=1e-6).to(device)
    liger_poly_norm = LigerPolyNorm(eps=1e-6).to(device).to(torch.float16)
    with torch.no_grad():
        liger_poly_norm.weight.copy_(naive_poly_norm.weight)
        liger_poly_norm.bias.copy_(naive_poly_norm.bias)
        ref_output = naive_poly_norm(x.float())
        triton_output = liger_poly_norm(x)

    assert torch.isfinite(triton_output).all()
    assert_verbose_allclose(ref_output, triton_output.float(), atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 2, 8),
        # weird shapes
        (9, 7, 41),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-6),
        pytest.param(
            torch.bfloat16,
            2e-1,
            2e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness_functional(bs, sl, hd, dtype, atol, rtol):
    """
    Test liger_poly_norm functional API correctness.

    Args:
        bs: batch size
        sl: sequence length
        hd: hidden dimension
        dtype: data type (float32 or bfloat16)
        atol: absolute tolerance
        rtol: relative tolerance
    """
    _tensor = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    x1 = _tensor.clone().requires_grad_(True)
    x2 = _tensor.clone().requires_grad_(True)

    weight = torch.tensor([0.3, 0.4, 0.3], device=device, dtype=dtype)
    bias = torch.tensor(0.1, device=device, dtype=dtype)

    weight1 = weight.clone().requires_grad_(True)
    bias1 = bias.clone().requires_grad_(True)

    weight2 = weight.clone().requires_grad_(True)
    bias2 = bias.clone().requires_grad_(True)

    # First call - functional API
    y1 = liger_poly_norm(x1, weight1, bias1, 1e-6)

    # Second call - Function.apply API (should be identical)
    y2 = LigerPolyNormFunction.apply(x2, weight2, bias2, 1e-6)

    # Check forward pass
    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    grad = torch.randn_like(y2)
    grad1 = grad.clone()
    grad2 = grad.clone()

    y1.backward(grad1)
    y2.backward(grad2)

    # Check gradients
    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
    assert torch.allclose(weight1.grad, weight2.grad, atol=atol, rtol=rtol)
    assert torch.allclose(bias1.grad, bias2.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize(
    "bs, sl, hd",
    [
        (2, 128, 512),
        (4, 256, 1024),
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        pytest.param(
            torch.bfloat16,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_forward_shapes(bs, sl, hd, dtype):
    """
    Test that LigerPolyNorm preserves input shapes correctly.

    Args:
        bs: batch size
        sl: sequence length
        hd: hidden dimension
        dtype: data type
    """
    x = torch.randn(bs, sl, hd, device=device, dtype=dtype)

    poly_norm = LigerPolyNorm(eps=1e-6).to(device).to(dtype)
    output = poly_norm(x)

    assert output.shape == x.shape, f"Output shape {output.shape} != input shape {x.shape}"
    assert output.dtype == x.dtype, f"Output dtype {output.dtype} != input dtype {x.dtype}"


@pytest.mark.parametrize(
    "shape",
    [
        (32, 512),  # 2D
        (8, 16, 512),  # 3D
        (4, 8, 16, 512),  # 4D
    ],
)
def test_multidimensional_input(shape):
    """
    Test that LigerPolyNorm handles multi-dimensional inputs correctly.

    Args:
        shape: input tensor shape
    """
    x = torch.randn(*shape, device=device, dtype=torch.float32, requires_grad=True)

    poly_norm = LigerPolyNorm(eps=1e-6).to(device)
    output = poly_norm(x)

    assert output.shape == shape, f"Output shape {output.shape} != input shape {shape}"

    # Test backward
    grad_output = torch.randn_like(output)
    output.backward(grad_output)

    assert x.grad is not None, "Gradient should be computed for input"
    assert x.grad.shape == shape, f"Gradient shape {x.grad.shape} != input shape {shape}"


@pytest.mark.parametrize(
    "n_rows, hd",
    [
        (4096, 16),
        (4103, 61),  # ragged row block + odd hidden size
        (32771, 123),
        (65537, 256),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-3, 1e-4),
        pytest.param(
            torch.bfloat16,
            2e-2,
            2e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
@pytest.mark.parametrize("in_place", [True, False])
def test_correctness_block_row(n_rows, hd, dtype, atol, rtol, in_place):
    """Exercise the block-row PolyNorm kernels (BLOCK_SIZE <= 256 and n_rows >= 4096)."""
    _tensor = torch.randn(n_rows, hd, device=device, dtype=dtype)
    grad_output = torch.randn(n_rows, hd, device=device, dtype=dtype)

    # Evaluate the reference in fp32 on the quantized inputs (matching the kernel's fp32 accumulation,
    # since summing millions of bf16 terms in bf16 drifts heavily due to mantissa truncation).
    x_ref = _tensor.detach().to(torch.float32).clone().requires_grad_(True)
    x_blk = _tensor.detach().clone().requires_grad_(True)

    blk_mod = LigerPolyNorm(eps=1e-6, in_place=in_place, row_mode=None).to(device).to(dtype)
    ref_mod = NaivePolyNorm(eps=1e-6).to(device)
    with torch.no_grad():
        ref_mod.weight.copy_(blk_mod.weight.float())
        ref_mod.bias.copy_(blk_mod.bias.float())

    y_ref = ref_mod(x_ref).to(dtype)
    y_blk = blk_mod(x_blk)
    y_ref.backward(grad_output.clone())
    y_blk.backward(grad_output.clone())

    assert_verbose_allclose(y_ref, y_blk, atol=atol, rtol=rtol)
    assert_verbose_allclose(x_ref.grad.to(dtype), x_blk.grad, atol=atol, rtol=rtol, max_print=20)
    assert_verbose_allclose(ref_mod.weight.grad.to(dtype), blk_mod.weight.grad, atol=atol, rtol=rtol)
    assert_verbose_allclose(ref_mod.bias.grad.to(dtype), blk_mod.bias.grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_poly_norm_bias_grad_preserves_bias_dtype():
    """dB must take B.dtype rather than W.dtype (avoiding rounding fp32 bias grads through bf16)."""
    rows, cols = 257, 16
    x = torch.ones(rows, cols, device=device, dtype=torch.bfloat16, requires_grad=True)
    w = torch.zeros(3, device=device, dtype=torch.bfloat16, requires_grad=True)
    b = torch.zeros((), device=device, dtype=torch.float32, requires_grad=True)

    y = liger_poly_norm(x, w, b, eps=1e-6, in_place=False)
    _, dw, db = torch.autograd.grad(y, (x, w, b), torch.ones_like(y))
    assert dw.dtype == torch.bfloat16
    assert db.dtype == torch.float32
    assert_verbose_allclose(db, torch.full_like(b, rows * cols))
