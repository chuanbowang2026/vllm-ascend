# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scatter per-token MXFP K scales into the PA_NZ K-scale cache.

Replaces the eager advanced-indexing assignment

    key_scale_cache[block_ids, :, seg_ids, :, frag_ids, :] = key_scale

whose three advanced indices sit on dims 0/2/4 separated by basic slices on
dims 1/3/5. Torch must broadcast the advanced indices into a leading result
axis and materialise the permuted intermediate before handing the write to
``IndexPutV2``; on float8 the operator falls back to AICPU entirely. The
Triton kernel computes every destination address directly, so no intermediate
is built and the write stays on AICore for any 1-byte scale dtype.

Layout contract (unchanged from the eager path):

    key_scale        [num_tokens, N, F, 2]            1-byte dtype, dense
    key_scale_cache  [num_blocks, N, S, F, 16, 2]     1-byte dtype, PA_NZ
    token at in-block offset ``o`` -> ``[block, n, o // 16, f, o % 16, j]``

``N`` is ``num_kv_heads``, ``F`` is ``head_dim // MXFP_KV_SCALE_GROUP_SIZE``,
``S`` is ``block_size // MXFP_K_SCALE_NZ_TOKEN_FRAG`` and the trailing axis
holds ``MXFP_KV_SCALE_VALUES_PER_GROUP`` bytes.

Both tensors end in a ``size=2, stride=1`` byte pair, so the launcher
bitcasts them to a 16-bit dtype and drops the ``j`` axis. That halves the
element count and doubles the access granularity; the kernel then performs a
plain ``dst[block, n, seg, f, frag] = src[t, n, f]``.

Padded rows arrive already clamped to slot 0 by ``mxfp_k_scale_slot_index``
and are written to block 0, the null block BlockPool never hands to a
request. The kernel preserves that behaviour rather than masking, which keeps
shapes static and the launch capture-safe: no host-device sync and no
data-dependent shapes.
"""

from __future__ import annotations

import torch
from vllm.triton_utils import tl, triton

# Target element count for one program's tile, in 16-bit elements. 4096
# elements is 8 KiB, comfortably inside the smallest shipping UB (128 KiB)
# while still giving each program enough work to amortise its launch.
_TILE_ELEMENTS = 4096
_MAX_BLOCK_T = 256


@triton.jit(do_not_specialize=["num_tokens"])
def _scatter_mxfp_k_scale_kernel(
    src_ptr,
    cache_ptr,
    block_ids_ptr,
    seg_ids_ptr,
    frag_ids_ptr,
    num_tokens,
    src_stride_t,
    src_stride_n,
    src_stride_f,
    cache_stride_b,
    cache_stride_n,
    cache_stride_s,
    cache_stride_f,
    cache_stride_g,
    block_ids_stride,
    seg_ids_stride,
    frag_ids_stride,
    F: tl.constexpr,
    PAYLOAD: tl.constexpr,
    PAYLOAD_P2: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Write one tile of ``BLOCK_T`` tokens x ``PAYLOAD`` scale elements.

    Offsets are in ELEMENTS of the bitcast 16-bit view, not bytes; the typed
    pointers carry the item size. Token indices are promoted to int64 before
    any address multiplication so a large cache cannot overflow a 32-bit
    intermediate.
    """
    token = (tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)).to(tl.int64)
    token_mask = token < num_tokens
    # Out-of-range lanes read index 0 and are masked out of the store; the
    # safe index keeps the gather itself in bounds.
    safe_token = tl.where(token_mask, token, 0)

    block = tl.load(block_ids_ptr + safe_token * block_ids_stride, mask=token_mask, other=0).to(tl.int64)
    seg = tl.load(seg_ids_ptr + safe_token * seg_ids_stride, mask=token_mask, other=0).to(tl.int64)
    frag = tl.load(frag_ids_ptr + safe_token * frag_ids_stride, mask=token_mask, other=0).to(tl.int64)

    # Flat payload index decomposes into (head, fragment). F is a constexpr,
    # so a power-of-two head dim compiles these to shifts.
    payload = tl.arange(0, PAYLOAD_P2).to(tl.int64)
    head = payload // F
    frag_dim = payload % F
    payload_mask = payload < PAYLOAD

    mask = token_mask[:, None] & payload_mask[None, :]

    src_offset = safe_token[:, None] * src_stride_t + head[None, :] * src_stride_n + frag_dim[None, :] * src_stride_f
    cache_offset = (
        block[:, None] * cache_stride_b
        + head[None, :] * cache_stride_n
        + seg[:, None] * cache_stride_s
        + frag_dim[None, :] * cache_stride_f
        + frag[:, None] * cache_stride_g
    )

    value = tl.load(src_ptr + src_offset, mask=mask, other=0)
    tl.store(cache_ptr + cache_offset, value, mask=mask)


def _bitcast_pairs(tensor: torch.Tensor) -> torch.Tensor:
    """View a trailing ``size=2, stride=1`` byte axis as one 16-bit element.

    Returns a tensor with the trailing axis dropped. Raises if the layout
    does not permit the bitcast, which lets the caller fall back.
    """
    if tensor.shape[-1] != 2 or tensor.stride(-1) != 1:
        raise ValueError("trailing scale axis must be 2 contiguous 1-byte values")
    if tensor.element_size() != 1:
        raise ValueError("scale tensors must have a 1-byte dtype")
    # view(dtype) validates the remaining strides are even, i.e. that every
    # axis still lands on a 16-bit boundary. A PA_NZ page gap that is not a
    # multiple of 2 bytes raises RuntimeError; surface every layout rejection
    # as ValueError so callers can distinguish "layout unsupported, fall back"
    # from a genuine kernel failure.
    try:
        return tensor.view(torch.int16).squeeze(-1)
    except RuntimeError as err:
        raise ValueError(f"scale tensor layout does not permit a 16-bit bitcast: {err}") from err


def scatter_mxfp_k_scale_cache_triton(
    key_scale: torch.Tensor,
    key_scale_cache: torch.Tensor,
    block_ids: torch.Tensor,
    seg_ids: torch.Tensor,
    frag_ids: torch.Tensor,
) -> None:
    """Scatter ``key_scale`` into ``key_scale_cache`` in one NPU launch.

    Semantically identical to the eager
    ``key_scale_cache[block_ids, :, seg_ids, :, frag_ids, :] = key_scale``,
    including the null-block destination for clamped padded rows. Destination
    slots must be unique, exactly as advanced-indexing assignment requires;
    uniqueness is a caller precondition and is not checked here because that
    would need a device-to-host sync.
    """
    num_tokens = key_scale.shape[0]
    if num_tokens == 0:
        return

    src = _bitcast_pairs(key_scale)
    cache = _bitcast_pairs(key_scale_cache)

    num_heads, num_frags = src.shape[1], src.shape[2]
    if cache.shape[1] != num_heads or cache.shape[3] != num_frags:
        raise ValueError(
            f"cache head/fragment axes {cache.shape[1]}/{cache.shape[3]} do not match source {num_heads}/{num_frags}"
        )

    payload = num_heads * num_frags
    # Plain arithmetic rather than triton.next_power_of_2 / triton.cdiv so the
    # launch geometry stays computable (and unit-testable) without a Triton
    # runtime present.
    payload_p2 = 1 << (payload - 1).bit_length()
    block_t = max(1, min(_MAX_BLOCK_T, _TILE_ELEMENTS // payload_p2))
    num_programs = (num_tokens + block_t - 1) // block_t

    _scatter_mxfp_k_scale_kernel[(num_programs,)](
        src,
        cache,
        block_ids,
        seg_ids,
        frag_ids,
        num_tokens,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        cache.stride(0),
        cache.stride(1),
        cache.stride(2),
        cache.stride(3),
        cache.stride(4),
        block_ids.stride(0),
        seg_ids.stride(0),
        frag_ids.stride(0),
        F=num_frags,
        PAYLOAD=payload,
        PAYLOAD_P2=payload_p2,
        BLOCK_T=block_t,
    )
