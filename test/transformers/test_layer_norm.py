import types

import pytest
import torch

from liger_kernel.ops import LigerLayerNormFunction
from liger_kernel.transformers.functional import liger_layer_norm
from liger_kernel.transformers.layer_norm import LigerLayerNorm
from liger_kernel.utils import infer_device

device = infer_device()


@pytest.mark.parametrize("bias", [True, False])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [
        (2, 8, 64),
        (4, 16, 128),
        (1, 1, 1023),  # Minimal batch/seq with near power-of-2 hidden
        (3, 7, 256),  # Prime numbers for batch/seq
        (1, 1, 1500),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 2e-2, 2e-2),  # Relaxed tolerance for bfloat16 due to lower precision + atomic limitations
    ],
)
def test_liger_layer_norm(
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    bias: bool,
) -> None:
    """Test basic layer norm functionality against PyTorch implementation."""
    torch.manual_seed(0)

    x = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)

    liger_x = x.clone().requires_grad_(True)
    torch_x = x.clone().requires_grad_(True)

    liger_ln = LigerLayerNorm(hidden_size, eps=1e-6, bias=bias).to(dtype).to(device)
    torch_ln = torch.nn.LayerNorm(hidden_size, eps=1e-6, bias=bias).to(dtype).to(device)

    with torch.no_grad():
        w = torch.randn(hidden_size, dtype=dtype, device=device)
        liger_ln.weight.copy_(w)
        torch_ln.weight.copy_(w)
        if bias:
            b = torch.randn(hidden_size, dtype=dtype, device=device)
            liger_ln.bias.copy_(b)
            torch_ln.bias.copy_(b)
        else:
            assert liger_ln.bias is None

    liger_output = liger_ln(liger_x)
    torch_output = torch_ln(torch_x)

    assert torch.allclose(liger_output, torch_output, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(x)
    liger_output.backward(grad_output, retain_graph=True)
    torch_output.backward(grad_output, retain_graph=True)

    assert torch.allclose(liger_x.grad, torch_x.grad, atol=atol, rtol=rtol)
    assert torch.allclose(liger_ln.weight.grad, torch_ln.weight.grad, atol=atol, rtol=rtol)
    if bias:
        assert torch.allclose(liger_ln.bias.grad, torch_ln.bias.grad, atol=atol, rtol=rtol)


@pytest.mark.parametrize("has_bias", [True, False])
@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size",
    [
        (2, 8, 64),
        (4, 16, 128),
        (3, 512, 128),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 2e-2, 2e-2),  # Relaxed tolerance for bfloat16 due to lower precision + atomic limitations
    ],
)
def test_liger_layer_norm_functional(
    hidden_size: int,
    batch_size: int,
    seq_len: int,
    dtype: torch.dtype,
    atol: float,
    rtol: float,
    has_bias: bool,
) -> None:
    """Test functional layer norm interface against autograd function."""
    torch.manual_seed(0)

    input = torch.randn(batch_size, seq_len, hidden_size, dtype=dtype, device=device)

    x1 = input.clone().requires_grad_(True)
    x2 = input.clone().requires_grad_(True)

    w = torch.randn(hidden_size, device=device, dtype=dtype)
    w1 = w.clone().requires_grad_(True)
    w2 = w.clone().requires_grad_(True)

    if has_bias:
        b = torch.randn(hidden_size, device=device, dtype=dtype)
        b1 = b.clone().requires_grad_(True)
        b2 = b.clone().requires_grad_(True)
    else:
        b1 = None
        b2 = None

    y1 = liger_layer_norm(X=x1, W=w1, B=b1, eps=1e-6)
    y2 = LigerLayerNormFunction.apply(x2, w2, b2, 1e-6)

    assert torch.allclose(y1, y2, atol=atol, rtol=rtol)

    grad_output = torch.randn_like(y2)
    y1.backward(grad_output, retain_graph=True)
    y2.backward(grad_output, retain_graph=True)

    assert torch.allclose(x1.grad, x2.grad, atol=atol, rtol=rtol)
    assert torch.allclose(w1.grad, w2.grad, atol=atol, rtol=rtol)
    if has_bias:
        assert torch.allclose(b1.grad, b2.grad, atol=atol, rtol=rtol)


def test_liger_layer_norm_bias_init_and_optimizer_parity() -> None:
    """Verify bias=True zero-initializes bias and bias=False registers None (no phantom bias updates)."""
    hidden_size = 64

    ln_with_bias = LigerLayerNorm(hidden_size, bias=True).to(device)
    torch_with_bias = torch.nn.LayerNorm(hidden_size, bias=True).to(device)
    assert ln_with_bias.bias is not None
    assert torch.equal(ln_with_bias.bias, torch.zeros_like(ln_with_bias.bias))
    assert set(ln_with_bias.state_dict().keys()) == set(torch_with_bias.state_dict().keys())

    ln_no_bias = LigerLayerNorm(hidden_size, bias=False).to(device)
    torch_no_bias = torch.nn.LayerNorm(hidden_size, bias=False).to(device)
    assert ln_no_bias.bias is None
    assert set(ln_no_bias.state_dict().keys()) == set(torch_no_bias.state_dict().keys()) == {"weight"}

    opt_liger = torch.optim.AdamW(ln_no_bias.parameters(), lr=1e-1)
    opt_torch = torch.optim.AdamW(torch_no_bias.parameters(), lr=1e-1)

    x = torch.randn(4, 8, hidden_size, device=device, dtype=torch.float32)
    scale = torch.linspace(-1.0, 1.0, hidden_size, device=device)
    (ln_no_bias(x) * scale).sum().backward()
    (torch_no_bias(x) * scale).sum().backward()
    opt_liger.step()
    opt_torch.step()

    assert ln_no_bias.bias is None
    assert torch.allclose(ln_no_bias(x), torch_no_bias(x), atol=1e-5, rtol=1e-5)


def test_liger_layer_norm_patched_biasless_torch_module() -> None:
    """Verify LigerLayerNorm.forward works when bound onto an existing nn.LayerNorm(bias=False)."""
    hidden_size = 64
    torch_ln = torch.nn.LayerNorm(hidden_size, eps=1e-6, bias=False).to(device)
    ref_ln = torch.nn.LayerNorm(hidden_size, eps=1e-6, bias=False).to(device)
    with torch.no_grad():
        w = torch.randn(hidden_size, device=device)
        torch_ln.weight.copy_(w)
        ref_ln.weight.copy_(w)

    # Mirror _patch_layer_norm_module attribute setup
    torch_ln.variance_epsilon = torch_ln.eps
    torch_ln.hidden_size = hidden_size
    torch_ln.impl = None
    torch_ln.mode = None
    torch_ln.forward = types.MethodType(LigerLayerNorm.forward, torch_ln)

    x1 = torch.randn(2, 8, hidden_size, device=device, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    y1 = torch_ln(x1)
    y2 = ref_ln(x2)
    assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-5)

    dy = torch.randn_like(y1)
    y1.backward(dy)
    y2.backward(dy)
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(torch_ln.weight.grad, ref_ln.weight.grad, atol=1e-5, rtol=1e-5)
