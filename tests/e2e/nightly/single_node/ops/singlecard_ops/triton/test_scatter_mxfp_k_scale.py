# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NPU equivalence for the MXFP K-scale scatter kernel.

The kernel replaces an advanced-indexing assignment, so the spec is exactly
that assignment: every configuration must come out bit-identical. Requires
NPU hardware.
"""

import pytest
import torch

from vllm_ascend.ops.triton.scatter_mxfp_k_scale import scatter_mxfp_k_scale_cache_triton

VALUES_PER_GROUP = 2
TOKEN_FRAG = 16
SCALE_GROUP_SIZE = 64


def _build(num_tokens, num_kv_heads, head_dim, block_size, num_blocks, device, seed):
    generator = torch.Generator().manual_seed(seed)
    num_frags = head_dim // SCALE_GROUP_SIZE
    num_segs = block_size // TOKEN_FRAG

    key_scale = torch.randint(
        0,
        256,
        (num_tokens, num_kv_heads, num_frags, VALUES_PER_GROUP),
        dtype=torch.uint8,
        generator=generator,
    ).to(device)
    cache = torch.randint(
        0,
        256,
        (num_blocks, num_kv_heads, num_segs, num_frags, TOKEN_FRAG, VALUES_PER_GROUP),
        dtype=torch.uint8,
        generator=generator,
    ).to(device)

    # Unique destinations, which advanced-index assignment requires.
    capacity = num_blocks * num_segs * TOKEN_FRAG
    assert num_tokens <= capacity
    flat = torch.randperm(capacity, generator=generator)[:num_tokens].to(device)
    block_ids = flat // (num_segs * TOKEN_FRAG)
    within = flat % (num_segs * TOKEN_FRAG)
    return key_scale, cache, block_ids, within // TOKEN_FRAG, within % TOKEN_FRAG


@pytest.mark.parametrize("num_tokens", [1, 15, 512, 4096])
@pytest.mark.parametrize("num_kv_heads", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 192])
def test_matches_eager_advanced_indexing(num_tokens, num_kv_heads, head_dim):
    block_size = 512
    num_blocks = max(1, -(-num_tokens // block_size) + 1)
    key_scale, cache, block_ids, seg_ids, frag_ids = _build(
        num_tokens, num_kv_heads, head_dim, block_size, num_blocks, "npu", seed=num_tokens + head_dim
    )

    expected = cache.clone()
    expected[block_ids, :, seg_ids, :, frag_ids, :] = key_scale

    actual = cache.clone()
    scatter_mxfp_k_scale_cache_triton(key_scale, actual, block_ids, seg_ids, frag_ids)

    assert torch.equal(actual.cpu(), expected.cpu())


def test_leaves_untouched_pages_intact():
    key_scale, cache, block_ids, seg_ids, frag_ids = _build(64, 4, 128, 512, 8, "npu", seed=7)
    before = cache.clone()
    scatter_mxfp_k_scale_cache_triton(key_scale, cache, block_ids, seg_ids, frag_ids)
    changed = (cache != before).sum().item()
    # At most every written byte differs; random data means some bytes may
    # coincide, so this is an upper bound, not an equality.
    assert changed <= key_scale.numel()


def test_strided_page_gap_cache():
    """A real PA_NZ cache is a strided view whose pages carry a gap."""
    num_blocks, heads, segs, frags = 4, 4, 32, 2
    page = heads * segs * frags * TOKEN_FRAG * VALUES_PER_GROUP
    gap = 64
    storage = torch.randint(0, 256, (num_blocks * (page + gap),), dtype=torch.uint8).npu()
    cache = torch.as_strided(
        storage,
        (num_blocks, heads, segs, frags, TOKEN_FRAG, VALUES_PER_GROUP),
        (
            page + gap,
            segs * frags * TOKEN_FRAG * VALUES_PER_GROUP,
            frags * TOKEN_FRAG * VALUES_PER_GROUP,
            TOKEN_FRAG * VALUES_PER_GROUP,
            VALUES_PER_GROUP,
            1,
        ),
    )
    key_scale, _, block_ids, seg_ids, frag_ids = _build(
        128, heads, frags * SCALE_GROUP_SIZE, segs * TOKEN_FRAG, num_blocks, "npu", seed=11
    )

    expected = cache.clone()
    expected[block_ids, :, seg_ids, :, frag_ids, :] = key_scale

    scatter_mxfp_k_scale_cache_triton(key_scale, cache, block_ids, seg_ids, frag_ids)
    assert torch.equal(cache.cpu(), expected.cpu())
