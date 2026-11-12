import pytest
import torch

from test.utils import supports_bfloat16

from liger_kernel.transformers.group_norm import LigerGroupNorm
from liger_kernel.utils import infer_device

device = infer_device()


def _test_liger_group_norm(shape, num_channels, num_groups, dtype, atol, rtol):
    torch.manual_seed(0)

    _tensor = torch.randn(*shape, dtype=dtype, device=device)

    liger_x = _tensor.clone().detach().requires_grad_(True)
    torch_x = _tensor.clone().detach().requires_grad_(True)

    liger_ln = LigerGroupNorm(num_channels, num_groups, eps=1e-6).to(dtype).to(device)
    torch_ln = torch.nn.GroupNorm(num_channels=num_channels, num_groups=num_groups, eps=1e-6).to(dtype).to(device)

    with torch.no_grad():
        torch_ln.weight.copy_(liger_ln.weight)
        torch_ln.bias.copy_(liger_ln.bias)

    liger_output = liger_ln(
        liger_x,
    )
    torch_output = torch_ln(torch_x)

    assert torch.allclose(liger_output, torch_output, atol=atol, rtol=rtol)
    grad_output = torch.randn_like(torch_x)
    liger_output.backward(grad_output, retain_graph=True)
    torch_output.backward(grad_output, retain_graph=True)
    assert torch.allclose(liger_x.grad, torch_x.grad, atol=atol, rtol=rtol)
    assert torch.allclose(liger_ln.bias.grad, torch_ln.bias.grad, atol=atol, rtol=rtol), "Bias grads different"
    assert torch.allclose(liger_ln.weight.grad, torch_ln.weight.grad, atol=atol, rtol=rtol), "Weight grads different"


@pytest.mark.parametrize(
    "batch_size, num_channels, num_groups, hidden_size",
    [
        (1, 1, 1, 3),  # minimal
        (1, 32, 32, 4),  # group == channel
        (16, 32, 1, 4096),  # single group
        (2, 63, 21, 2163),  # non-aligned hidden
        (16, 48, 12, 8192),  # large hidden
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_liger_group_norm(batch_size, num_channels, num_groups, hidden_size, dtype, atol, rtol):
    _test_liger_group_norm((batch_size, num_channels, hidden_size), num_channels, num_groups, dtype, atol, rtol)


@pytest.mark.parametrize(
    "shape, num_channels, num_groups",
    [
        ((2, 6, 4, 8), 6, 3),  # convolutional features
        ((4, 32, 16, 16), 32, 8),  # larger spatial plane
        ((2, 8, 2, 4, 4), 8, 4),  # volumetric features
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
    ],
)
def test_liger_group_norm_spatial_dims(shape, num_channels, num_groups, dtype, atol, rtol):
    _test_liger_group_norm(shape, num_channels, num_groups, dtype, atol, rtol)


@pytest.mark.parametrize(
    "shape, num_groups, offset, dtype, atol, rtol",
    [
        # fp16: sum(x * x) over a 65536-element group exceeds the fp16 max (65504)
        ((1, 512, 64, 64), 32, 0.0, torch.float16, 2e-3, 2e-3),
        ((1, 512, 64, 64), 32, 1.0, torch.float16, 2e-3, 2e-3),
        # bf16: E[x^2] - E[x]^2 cancels when the mean is large relative to the spread
        pytest.param(
            (1, 64, 8),
            8,
            16.0,
            torch.bfloat16,
            1e-2,
            1e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_liger_group_norm_low_precision_statistics(shape, num_groups, offset, dtype, atol, rtol):
    # the group mean and variance must be accumulated in fp32 even for fp16/bf16 inputs
    torch.manual_seed(0)
    x = (torch.randn(*shape, device=device) + offset).to(dtype)
    num_channels = shape[1]

    liger_gn = LigerGroupNorm(num_channels, num_groups, eps=1e-6).to(dtype).to(device)
    with torch.no_grad():
        liger_output = liger_gn(x)
        torch_output = torch.nn.functional.group_norm(
            x.float(), num_groups, liger_gn.weight.float(), liger_gn.bias.float(), eps=1e-6
        )

    assert torch.isfinite(liger_output).all()
    assert torch.allclose(liger_output.float(), torch_output, atol=atol, rtol=rtol)


def test_liger_group_norm_2d_and_repr():
    gn = LigerGroupNorm(8, 2, eps=1e-6).to(device)
    assert "num_channels=8" in repr(gn) and "num_groups=2" in repr(gn)

    x1 = torch.randn(4, 8, device=device, dtype=torch.float32, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    ref = torch.nn.GroupNorm(2, 8, eps=1e-6).to(device)
    with torch.no_grad():
        ref.weight.copy_(gn.weight)
        ref.bias.copy_(gn.bias)

    y1 = gn(x1)
    y2 = ref(x2)
    assert torch.allclose(y1, y2, atol=1e-5, rtol=1e-5)
    dy = torch.randn_like(y1)
    y1.backward(dy)
    y2.backward(dy.clone())
    assert torch.allclose(x1.grad, x2.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-4, 1e-4),
        (torch.float16, 1e-2, 1e-2),
        pytest.param(
            torch.bfloat16,
            2e-2,
            2e-2,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
@pytest.mark.parametrize("channels_last", [False, True])
def test_liger_group_norm_large_spatial_and_channels_last(dtype, atol, rtol, channels_last):
    """Exercise split-spatial NCHW and native 4D channels_last forward + backward paths."""
    torch.manual_seed(0)
    shape = (2, 24, 1, 16389)  # non-power-of-2 spatial length >= 4096, small N*G
    num_channels, num_groups = 24, 8

    x_base = (torch.randn(*shape, device=device) + 8.0).to(dtype)
    dy_base = torch.randn(*shape, device=device, dtype=dtype)
    if channels_last:
        x_base = x_base.contiguous(memory_format=torch.channels_last)
        dy_base = dy_base.contiguous(memory_format=torch.channels_last)

    x_lig = x_base.clone().requires_grad_(True)
    x_ref = x_base.detach().to(torch.float32).clone().requires_grad_(True)

    gn_lig = LigerGroupNorm(num_channels, num_groups, eps=1e-5).to(device).to(dtype)
    w_ref = gn_lig.weight.detach().to(torch.float32).clone().requires_grad_(True)
    b_ref = gn_lig.bias.detach().to(torch.float32).clone().requires_grad_(True)

    y_lig = gn_lig(x_lig)
    if channels_last:
        assert y_lig.is_contiguous(memory_format=torch.channels_last)
    y_ref = torch.nn.functional.group_norm(x_ref, num_groups, w_ref, b_ref, eps=1e-5)

    assert torch.isfinite(y_lig).all()
    assert torch.allclose(y_lig.float(), y_ref, atol=atol, rtol=rtol)

    y_lig.backward(dy_base)
    y_ref.backward(dy_base.float())

    if channels_last:
        assert x_lig.grad.is_contiguous(memory_format=torch.channels_last)
    assert torch.isfinite(x_lig.grad).all()
    assert torch.allclose(x_lig.grad.float(), x_ref.grad, atol=atol, rtol=rtol)
    assert torch.allclose(
        gn_lig.weight.grad.float(), w_ref.grad, atol=5e-2 if dtype != torch.float32 else 1e-3, rtol=2e-2
    )
    if dtype != torch.float16:  # sum of 32k elements can exceed fp16 range on bias grad
        assert torch.allclose(
            gn_lig.bias.grad.float(), b_ref.grad, atol=5e-2 if dtype != torch.float32 else 1e-3, rtol=2e-2
        )
