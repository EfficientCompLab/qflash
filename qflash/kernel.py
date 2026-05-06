import os
import torch
from torch import Tensor

import triton
import triton.language as tl


KEYS = ['N_CTX', 'HEAD_DIM']


def get_fast_config():
    configs = []
    for BM in [32, 64]:
        for BN in [32, 64]:
            if BM <= BN:
                continue
            for w in [2, 4]:
                for s in [2, 4]:
                    configs.append(
                        triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_warps=w, num_stages=s)
                    )
    return configs


def get_default_config():
    return [triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=1, num_stages=1)]


AUTOTUNE_SETTING = os.environ.get("QFLASH_AUTOTUNE", "default")
USE_CUDA_GRAPH = os.environ.get("QFLASH_CUDA_GRAPH", "0") == "1"
EXP_USE_SHIFT = os.environ.get("QFLASH_EXP_USE_SHIFT", "1") == "1"
L_USE_SHIFT = os.environ.get("QFLASH_L_USE_SHIFT", "1") == "1"
ACC_USE_SHIFT = os.environ.get("QFLASH_ACC_USE_SHIFT", "1") == "1"
L_SHIFT = int(os.environ.get("QFLASH_L_SHIFT", "30"))
ACC_SHIFT = int(os.environ.get("QFLASH_ACC_SHIFT", "30"))
get_autotune_config = get_fast_config if AUTOTUNE_SETTING == "fast" else get_default_config


@triton.jit
def exp2_i32_shift(x, x_rscale, exp_M, SHIFT: tl.constexpr = 16):
    q = ((x * exp_M) >> SHIFT).to(tl.int32)
    r = (x + q * x_rscale).to(tl.int32)
    return (((r >> 1) + x_rscale) >> q).to(tl.int32)


@triton.jit
def exp2_i32_idiv(x, x_rscale):
    q = ((-x) // x_rscale).to(tl.int32)
    r = (x + q * x_rscale).to(tl.int32)
    return (((r >> 1) + x_rscale) >> q).to(tl.int32)


@triton.jit
def exp2_i32(x, x_rscale, exp_M, SHIFT: tl.constexpr = 16, USE_SHIFT: tl.constexpr = True):
    if USE_SHIFT:
        return exp2_i32_shift(x, x_rscale, exp_M, SHIFT)
    else:
        return exp2_i32_idiv(x, x_rscale)


@triton.jit
def requantize(x, req_M, SHIFT: tl.constexpr = 16):
    return ((x * req_M + (1 << (SHIFT - 1))) >> SHIFT).to(tl.int8)


@triton.jit
def scale_release_shift(x, alpha, M, SHIFT: tl.constexpr = 16):
    return ((x.to(tl.int64) * alpha * M + (1 << (SHIFT - 1))) >> SHIFT).to(tl.int32)


@triton.jit
def scale_release_idiv(x, alpha, sm_rscale):
    return ((x * alpha) // sm_rscale).to(tl.int32)


@triton.jit
def scale_release(x, alpha, sm_rscale, M, SHIFT: tl.constexpr = 16, USE_SHIFT: tl.constexpr = True):
    if USE_SHIFT:
        return scale_release_shift(x, alpha, M, SHIFT)
    else:
        return scale_release_idiv(x, alpha, sm_rscale)


@triton.autotune(configs=get_autotune_config(), key=KEYS, use_cuda_graph=USE_CUDA_GRAPH)
@triton.jit
def qflash_kernel(
    Q, K, V, Out,
    sm_rscale, exp_M, req_M, l_M, acc_M,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    Z, H,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXP_USE_SHIFT: tl.constexpr = True,
    L_USE_SHIFT: tl.constexpr = True,
    ACC_USE_SHIFT: tl.constexpr = True,
    L_SHIFT: tl.constexpr = 16,
    ACC_SHIFT: tl.constexpr = 16,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    EVEN_CTX: tl.constexpr = (N_CTX % BLOCK_M == 0) and (N_CTX % BLOCK_N == 0)

    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn), offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn), offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    NEG_INF = -(1 << 20)
    m_i = tl.zeros([BLOCK_M], dtype=tl.int32) + NEG_INF
    l_i = tl.zeros([BLOCK_M], dtype=tl.int32) + 1
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.int32)

    if EVEN_CTX:
        q = tl.load(Q_block_ptr)
    else:
        q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        if EVEN_CTX:
            k = tl.load(K_block_ptr)
        else:
            k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        qk = tl.dot(q, k, out_dtype=tl.int32)
        if not EVEN_CTX:
            padding = (offs_m < N_CTX)[:, None] & ((start_n + offs_n) < N_CTX)[None, :]
            qk = tl.where(padding, qk, NEG_INF)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = exp2_i32(qk, sm_rscale, exp_M, USE_SHIFT=EXP_USE_SHIFT)
        p = requantize(p, req_M)
        l_ij = tl.sum(p, 1)
        alpha = exp2_i32(m_i - m_ij, sm_rscale, exp_M, USE_SHIFT=EXP_USE_SHIFT)
        l_i = scale_release(l_i, alpha, sm_rscale, l_M, SHIFT=L_SHIFT, USE_SHIFT=L_USE_SHIFT) + l_ij
        acc = scale_release(acc, alpha[:, None], sm_rscale, acc_M, SHIFT=ACC_SHIFT, USE_SHIFT=ACC_USE_SHIFT)
        if EVEN_CTX:
            v = tl.load(V_block_ptr)
        else:
            v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        p = p.to(tl.int8)
        acc = tl.dot(p, v, acc, out_dtype=tl.int32)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))

    acc = (acc / l_i[:, None]).to(tl.int32)
    if EVEN_CTX:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty))
    else:
        tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,))


@triton.autotune(configs=get_autotune_config(), key=KEYS, use_cuda_graph=USE_CUDA_GRAPH)
@triton.jit
def qflash_masked_kernel(
    Q, K, V, Out, attn_bias,
    sm_rscale, exp_M, req_M, l_M, acc_M,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vk, stride_vn,
    stride_oz, stride_oh, stride_om, stride_on,
    stride_amz, stride_amh, stride_amm, stride_amn,
    Z, H,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EXP_USE_SHIFT: tl.constexpr = True,
    L_USE_SHIFT: tl.constexpr = True,
    ACC_USE_SHIFT: tl.constexpr = True,
    L_SHIFT: tl.constexpr = 16,
    ACC_SHIFT: tl.constexpr = 16,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    am_offset = off_z.to(tl.int64) * stride_amz + off_h.to(tl.int64) * stride_amh

    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_vk, stride_vn), offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM), order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset, shape=(HEAD_DIM, N_CTX),
        strides=(stride_kk, stride_kn), offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N), order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset, shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM), order=(1, 0),
    )
    AM_block_ptr = tl.make_block_ptr(
        base=attn_bias + am_offset, shape=(N_CTX, N_CTX),
        strides=(stride_amm, stride_amn), offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    NEG_INF = -(1 << 20)
    m_i = tl.zeros([BLOCK_M], dtype=tl.int32) + NEG_INF
    l_i = tl.zeros([BLOCK_M], dtype=tl.int32) + 1
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.int32)

    q = tl.load(Q_block_ptr, boundary_check=(0,), padding_option="zero")

    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k = tl.load(K_block_ptr, boundary_check=(1,), padding_option="zero")
        bias = tl.load(AM_block_ptr, boundary_check=(0, 1), padding_option="zero")
        qk = tl.dot(q, k, out_dtype=tl.int32)
        padding = (offs_m < N_CTX)[:, None] & ((start_n + offs_n) < N_CTX)[None, :]
        qk = tl.where(padding, qk + bias, NEG_INF)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]
        p = exp2_i32(qk, sm_rscale, exp_M, USE_SHIFT=EXP_USE_SHIFT)
        p = requantize(p, req_M)
        l_ij = tl.sum(p, 1)
        alpha = exp2_i32(m_i - m_ij, sm_rscale, exp_M, USE_SHIFT=EXP_USE_SHIFT)
        l_i = scale_release(l_i, alpha, sm_rscale, l_M, SHIFT=L_SHIFT, USE_SHIFT=L_USE_SHIFT) + l_ij
        acc = scale_release(acc, alpha[:, None], sm_rscale, acc_M, SHIFT=ACC_SHIFT, USE_SHIFT=ACC_USE_SHIFT)
        v = tl.load(V_block_ptr, boundary_check=(0,), padding_option="zero")
        p = p.to(tl.int8)
        acc = tl.dot(p, v, acc, out_dtype=tl.int32)
        m_i = m_ij
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_N))
        AM_block_ptr = tl.advance(AM_block_ptr, (0, BLOCK_N))

    acc = (acc / l_i[:, None]).to(tl.int32)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty), boundary_check=(0,))


def qflash_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn_mask: Tensor | None = None,
    scale: float | None = None,
    return_mode: str = 'output',
):
    HEAD_DIM_Q, HEAD_DIM_K = query.shape[-1], key.shape[-1]
    HEAD_DIM_V = value.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}
    LOG2E = 1.44269504
    NEG_INF = -(1 << 20)

    if scale is None:
        scale = 1.0 / (HEAD_DIM_Q ** 0.5)
    scale = scale * LOG2E

    q_scale = query.abs().amax() / torch.iinfo(torch.int8).max
    k_scale = key.abs().amax() / torch.iinfo(torch.int8).max
    o_scale = value.abs().amax() / torch.iinfo(torch.int8).max
    qk_scale = (q_scale * k_scale).item()
    sm_scale = scale * qk_scale

    # fixed-point bits: host scales by (1 << SHIFT), kernels do >> SHIFT.
    SHIFT = 16
    sm_rscale = int(round(1.0 / sm_scale))
    exp_M = int(round(-sm_scale * (1 << SHIFT)))
    req_M = int(round(sm_scale * 127 * (1 << SHIFT)))
    # reciprocals of sm_rscale for the scale_release shift path
    l_M = int(round((1 << L_SHIFT) / sm_rscale))
    acc_M = int(round((1 << ACC_SHIFT) / sm_rscale))

    query = torch.clamp(torch.round(query / q_scale), -128, 127).to(torch.int8)
    key = torch.clamp(torch.round(key / k_scale), -128, 127).to(torch.int8)
    value = torch.clamp(torch.round(value / o_scale), -128, 127).to(torch.int8)
    o = torch.empty_like(value)

    Z, H, N = query.shape[0], query.shape[1], query.shape[2]
    grid = lambda args: (triton.cdiv(N, args["BLOCK_M"]), Z * H, 1)

    if attn_mask is None:
        def kernel():
            qflash_kernel[grid](
                query, key, value, o,
                sm_rscale, exp_M, req_M, l_M, acc_M,
                *query.stride(), *key.stride(), *value.stride(), *o.stride(),
                Z, H, N_CTX=N, HEAD_DIM=HEAD_DIM_K,
                EXP_USE_SHIFT=EXP_USE_SHIFT, L_USE_SHIFT=L_USE_SHIFT, ACC_USE_SHIFT=ACC_USE_SHIFT,
                L_SHIFT=L_SHIFT, ACC_SHIFT=ACC_SHIFT,
            )
    else:
        if attn_mask.dtype == torch.bool:
            attn_bias = torch.zeros(Z, H, N, N, dtype=torch.int32, device=query.device)
            attn_bias.masked_fill_(attn_mask.logical_not(), NEG_INF)
        else:
            # bias * log2(e) / sm_scale, with mask-out positions overridden to NEG_INF.
            is_masked = attn_mask < -1e2
            clipped = attn_mask.masked_fill(is_masked, 0.0)
            quant = (clipped * LOG2E / sm_scale).round().to(torch.int32)
            attn_bias = quant.masked_fill(is_masked, NEG_INF).expand(Z, H, N, N).contiguous()

        def kernel():
            qflash_masked_kernel[grid](
                query, key, value, o, attn_bias,
                sm_rscale, exp_M, req_M, l_M, acc_M,
                *query.stride(), *key.stride(), *value.stride(), *o.stride(), *attn_bias.stride(),
                Z, H, N_CTX=N, HEAD_DIM=HEAD_DIM_K,
                EXP_USE_SHIFT=EXP_USE_SHIFT, L_USE_SHIFT=L_USE_SHIFT, ACC_USE_SHIFT=ACC_USE_SHIFT,
                L_SHIFT=L_SHIFT, ACC_SHIFT=ACC_SHIFT,
            )

    if return_mode == 'latency':
        return triton.testing.do_bench(kernel, warmup=10, rep=100, return_mode='median')
    kernel()
    return o * o_scale


class qflash:
    kernel = [qflash_kernel, qflash_masked_kernel]
    forward = qflash_forward
