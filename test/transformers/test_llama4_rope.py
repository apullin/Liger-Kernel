import pytest
import torch

from test.utils import supports_bfloat16

from liger_kernel.ops import LigerLlama4RopeFunction
from liger_kernel.transformers.llama4_rope import liger_llama4_text_rotary_pos_emb
from liger_kernel.transformers.llama4_rope import liger_llama4_vision_rotary_pos_emb
from liger_kernel.utils import get_total_gpu_memory
from liger_kernel.utils import infer_device

try:
    from transformers.models.llama4.configuration_llama4 import Llama4TextConfig
    from transformers.models.llama4.modeling_llama4 import Llama4TextRotaryEmbedding
    from transformers.models.llama4.modeling_llama4 import apply_rotary_emb

    IS_LLAMA4_AVAILABLE = True
except Exception:
    IS_LLAMA4_AVAILABLE = False

device = infer_device()


@pytest.mark.skipif(not IS_LLAMA4_AVAILABLE, reason="Llama4 is not available in transformers.")
@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 128, 32, 32, 64),
        (2, 128, 32, 32, 64),
        (1, 128, 32, 8, 64),
        (2, 128, 32, 8, 64),
        # weird shapes
        (3, 423, 73, 213, 92),
        (3, 423, 73, 155, 92),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-5,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_correctness(bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol):
    config = Llama4TextConfig(
        hidden_size=num_q_heads * head_dim,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=seq_len,
        rope_theta=10000.0,
        rope_scaling=None,
    )
    rotary_emb = Llama4TextRotaryEmbedding(config=config)

    _tensor_q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device).to(dtype)
    _tensor_k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device).to(dtype)

    q1 = _tensor_q.clone().requires_grad_(True)
    k1 = _tensor_k.clone().requires_grad_(True)
    q2 = _tensor_q.clone().requires_grad_(True)
    k2 = _tensor_k.clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    freqs_cis = rotary_emb(q1, pos_ids)

    hf_q, hf_k = apply_rotary_emb(q1, k1, freqs_cis)
    tt_q, tt_k = liger_llama4_text_rotary_pos_emb(q2, k2, freqs_cis)
    assert torch.allclose(hf_q, tt_q, atol=atol, rtol=rtol)
    assert torch.allclose(hf_k, tt_k, atol=atol, rtol=rtol)

    # backward
    dq, dk = torch.randn_like(hf_q, device=device), torch.randn_like(hf_k, device=device).to(dtype)

    q1_grad, k1_grad = torch.autograd.grad((hf_q, hf_k), (q1, k1), (dq, dk), allow_unused=True)
    q2_grad, k2_grad = torch.autograd.grad((tt_q, tt_k), (q2, k2), (dq.clone(), dk.clone()), allow_unused=True)

    assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)


@pytest.mark.skipif(not IS_LLAMA4_AVAILABLE, reason="Llama4 is not available in transformers.")
@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        (1, 2, 2, 2, 8),
        (1, 2, 1, 2, 8),
        (9, 7, 41, 41, 40),
    ],
)
@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        (torch.bfloat16, 1e-1, 1e-5),
    ],
)
def test_functional_correctness(bsz, seq_len, num_q_heads, num_kv_heads, head_dim, dtype, atol, rtol):
    config = Llama4TextConfig(
        hidden_size=num_q_heads * head_dim,
        num_attention_heads=num_q_heads,
        num_key_value_heads=num_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=seq_len,
        rope_theta=10000.0,
        rope_scaling=None,
    )
    rotary_emb = Llama4TextRotaryEmbedding(config=config)

    _q = torch.randn((bsz, seq_len, num_q_heads, head_dim), device=device, dtype=dtype)
    _k = torch.randn((bsz, seq_len, num_kv_heads, head_dim), device=device, dtype=dtype)

    q1 = _q.clone().requires_grad_(True)
    q2 = _q.clone().requires_grad_(True)
    k1 = _k.clone().requires_grad_(True)
    k2 = _k.clone().requires_grad_(True)

    pos_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(bsz, -1)
    freqs_cis = rotary_emb(q1, pos_ids)

    functional_q, functional_k = liger_llama4_text_rotary_pos_emb(q1, k1, freqs_cis)
    class_q, class_k = LigerLlama4RopeFunction.apply(q2, k2, freqs_cis)

    assert torch.allclose(functional_q, class_q, atol=atol, rtol=rtol)
    assert torch.allclose(functional_k, class_k, atol=atol, rtol=rtol)

    dq, dk = torch.randn_like(functional_q), torch.randn_like(functional_k)
    dq1, dk1 = dq.clone(), dk.clone()
    dq2, dk2 = dq.clone(), dk.clone()

    q1_grad, k1_grad = torch.autograd.grad(
        (functional_q, functional_k),
        (q1, k1),
        (dq1, dk1),
        allow_unused=True,
    )

    q2_grad, k2_grad = torch.autograd.grad(
        (class_q, class_k),
        (q2, k2),
        (dq2, dk2),
        allow_unused=True,
    )

    assert torch.allclose(q1_grad, q2_grad, atol=atol, rtol=rtol)
    assert torch.allclose(k1_grad, k2_grad, atol=atol, rtol=rtol)


# The kernel indexes the flattened batch*seq dimension as
# `base_offset * q_row_stride`, whose largest value is
# `(bsz * seq_len - 1) * n_q_heads * head_dim`. That product is one row short of
# `q.numel()`, so it only exceeds int32 once q itself holds more than 2**31
# elements. There is no cheaper shape: the bound is the element count.
_HEAD_DIM = 128
_N_Q_HEADS = 64
_ROW = _N_Q_HEADS * _HEAD_DIM
_SEQ_LEN = (2**31 - 1) // _ROW + 2  # smallest seq_len whose last offset wraps int32


@pytest.mark.skipif(not IS_LLAMA4_AVAILABLE, reason="Llama4 is not available in transformers.")
@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
@pytest.mark.parametrize(
    "bsz, seq_len, num_q_heads, num_kv_heads, head_dim",
    [
        pytest.param(
            1,
            _SEQ_LEN,
            _N_Q_HEADS,
            1,
            _HEAD_DIM,
            marks=pytest.mark.skipif(
                infer_device() == "cpu" or get_total_gpu_memory() < 20,
                reason="This test requires a GPU with at least 20GB of memory",
            ),
        ),
    ],
)
def test_row_offset_does_not_wrap_int32(bsz, seq_len, num_q_heads, num_kv_heads, head_dim):
    """Regression for #1335: the last token must still be rotated, not read from a wrapped address."""
    dtype = torch.bfloat16
    n_kv_heads = num_kv_heads

    q = torch.zeros((bsz, seq_len, num_q_heads, head_dim), device=device, dtype=dtype)
    k = torch.zeros((bsz, seq_len, n_kv_heads, head_dim), device=device, dtype=dtype)

    # Only the final token carries a signal. It sits at the offset that wraps.
    q[0, -1].fill_(1.0)
    k[0, -1].fill_(1.0)

    config = Llama4TextConfig(
        hidden_size=num_q_heads * head_dim,
        num_attention_heads=num_q_heads,
        num_key_value_heads=n_kv_heads,
        head_dim=head_dim,
        max_position_embeddings=seq_len,
        rope_theta=10000.0,
        rope_scaling=None,
    )
    rotary_emb = Llama4TextRotaryEmbedding(config=config)
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    freqs_cis = rotary_emb(q, pos_ids)

    # The kernel rotates q/k in place, so keep an unrotated copy of the last token for the reference.
    q_last = q[:, -1:].float().clone()
    k_last = k[:, -1:].float().clone()

    q_out, k_out = LigerLlama4RopeFunction.apply(q, k, freqs_cis)

    # A wrapped index leaves the last row untouched or fills it from elsewhere.
    assert torch.isfinite(q_out[0, -1]).all()
    assert torch.isfinite(k_out[0, -1]).all()
    assert q_out[0, -1].abs().sum() > 0
    assert k_out[0, -1].abs().sum() > 0

    ref_q, ref_k = apply_rotary_emb(q_last, k_last, freqs_cis[:, -1:])
    assert torch.allclose(q_out[:, -1:].float(), ref_q, atol=1e-1, rtol=1e-5)
    assert torch.allclose(k_out[:, -1:].float(), ref_k, atol=1e-1, rtol=1e-5)


def _reference_complex_rope(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_c = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_c = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    if freqs_cis.ndim == 2:
        freqs = freqs_cis[None, :, None, :]
    elif freqs_cis.ndim == 3:
        freqs = freqs_cis[:, :, None, :]
    elif freqs_cis.ndim == 4:
        freqs = freqs_cis
    else:
        raise ValueError(f"Unexpected shape: {freqs_cis.shape}")
    xq_out = torch.view_as_real(xq_c * freqs).flatten(3).type_as(xq)
    xk_out = torch.view_as_real(xk_c * freqs).flatten(3).type_as(xk)
    return xq_out, xk_out


@pytest.mark.parametrize(
    "dtype, atol, rtol",
    [
        (torch.float32, 1e-5, 1e-5),
        pytest.param(
            torch.bfloat16,
            1e-1,
            1e-5,
            marks=pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU"),
        ),
    ],
)
def test_per_batch_freqs_cis(dtype, atol, rtol):
    """Distinct per-batch RoPE frequencies (e.g. left-padded or KV-cache position_ids) must be respected."""
    bsz, seq_len, n_qh, n_kh, head_dim = 4, 9, 8, 2, 64
    q_base = torch.randn(bsz, seq_len, n_qh, head_dim, device=device, dtype=dtype)
    k_base = torch.randn(bsz, seq_len, n_kh, head_dim, device=device, dtype=dtype)
    angles = torch.randn(bsz, seq_len, head_dim // 2, device=device, dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    q_ref_in = q_base.clone().requires_grad_(True)
    k_ref_in = k_base.clone().requires_grad_(True)
    q_lig_in = q_base.clone().requires_grad_(True)
    k_lig_in = k_base.clone().requires_grad_(True)

    q_ref, k_ref = _reference_complex_rope(q_ref_in, k_ref_in, freqs_cis)
    q_lig, k_lig = liger_llama4_text_rotary_pos_emb(q_lig_in, k_lig_in, freqs_cis)

    assert torch.allclose(q_ref, q_lig, atol=atol, rtol=rtol)
    assert torch.allclose(k_ref, k_lig, atol=atol, rtol=rtol)

    dq = torch.randn_like(q_ref)
    dk = torch.randn_like(k_ref)
    dq_ref, dk_ref = torch.autograd.grad((q_ref, k_ref), (q_ref_in, k_ref_in), (dq, dk))
    dq_lig, dk_lig = torch.autograd.grad((q_lig, k_lig), (q_lig_in, k_lig_in), (dq.clone(), dk.clone()))

    assert torch.allclose(dq_ref, dq_lig, atol=atol, rtol=rtol)
    assert torch.allclose(dk_ref, dk_lig, atol=atol, rtol=rtol)


@pytest.mark.parametrize("bsz", [1, 4])
def test_seqlen1_transposed_view(bsz):
    """A [B, H, 1, D].transpose(1, 2) decode view is contiguous with stride(1) == D; token pitch is stride(0)."""
    n_qh, n_kh, head_dim = 8, 2, 64
    q_bhsd = torch.randn(bsz, n_qh, 1, head_dim, device=device, dtype=torch.float32, requires_grad=True)
    k_bhsd = torch.randn(bsz, n_kh, 1, head_dim, device=device, dtype=torch.float32, requires_grad=True)
    q_ref_bhsd = q_bhsd.detach().clone().requires_grad_(True)
    k_ref_bhsd = k_bhsd.detach().clone().requires_grad_(True)

    angles = torch.randn(bsz, 1, head_dim // 2, device=device, dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    q_t = q_bhsd.transpose(1, 2)
    k_t = k_bhsd.transpose(1, 2)
    assert q_t.is_contiguous() and q_t.stride(1) == head_dim

    q_ref, k_ref = _reference_complex_rope(q_ref_bhsd.transpose(1, 2), k_ref_bhsd.transpose(1, 2), freqs_cis)
    q_lig, k_lig = liger_llama4_text_rotary_pos_emb(q_t, k_t, freqs_cis)

    assert torch.allclose(q_ref, q_lig, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_ref, k_lig, atol=1e-5, rtol=1e-5)

    dq = torch.randn_like(q_ref)
    dk = torch.randn_like(k_ref)
    torch.autograd.backward((q_ref, k_ref), (dq, dk))
    torch.autograd.backward((q_lig, k_lig), (dq.clone(), dk.clone()))

    assert torch.allclose(q_ref_bhsd.grad, q_bhsd.grad, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_ref_bhsd.grad, k_bhsd.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("freqs_shape", [(6, 32), (1, 6, 32), (3, 6, 32), (1, 6, 1, 32), (3, 6, 1, 32)])
def test_vision_rotary_pos_emb_shapes(freqs_shape):
    bsz, seq_len, n_heads, head_dim = 3, 6, 4, 64
    q = torch.randn(bsz, seq_len, n_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)
    k = torch.randn(bsz, seq_len, n_heads, head_dim, device=device, dtype=torch.float32, requires_grad=True)
    angles = torch.randn(*freqs_shape, device=device, dtype=torch.float32)
    freqs_ci = torch.polar(torch.ones_like(angles), angles)

    q_ref, k_ref = _reference_complex_rope(q.detach().clone(), k.detach().clone(), freqs_ci)
    q_lig, k_lig = liger_llama4_vision_rotary_pos_emb(q, k, freqs_ci)

    assert torch.allclose(q_ref, q_lig, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_ref, k_lig, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not supports_bfloat16(), reason="bfloat16 not supported on this GPU")
def test_mixed_qk_dtypes_preserved():
    q = torch.randn(2, 4, 4, 32, device=device, dtype=torch.float32)
    k = torch.randn(2, 4, 2, 32, device=device, dtype=torch.bfloat16)
    angles = torch.randn(2, 4, 16, device=device, dtype=torch.float32)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)

    q_out, k_out = liger_llama4_text_rotary_pos_emb(q.clone(), k.clone(), freqs_cis)
    q_ref, k_ref = _reference_complex_rope(q, k, freqs_cis)
    assert q_out.dtype == torch.float32
    assert k_out.dtype == torch.bfloat16
    assert torch.allclose(q_out, q_ref, atol=1e-5, rtol=1e-5)
    assert torch.allclose(k_out, k_ref, atol=1e-1, rtol=1e-5)
