# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch geometry and address arithmetic for the MXFP K-scale scatter.

The kernel itself needs an NPU, so these tests pin the two things that are
device independent and easiest to get wrong: the bitcast contract, and the
flat address formula the kernel uses. The formula is checked against the
eager advanced-indexing assignment it replaces, which is the real spec.
Numerical equivalence of the compiled kernel is covered by the nightly NPU
test.
"""

from unittest.mock import patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.ops.triton.scatter_mxfp_k_scale import (
    _bitcast_pairs,
    scatter_mxfp_k_scale_cache_triton,
)

VALUES_PER_GROUP = 2
TOKEN_FRAG = 16


def _make_case(num_tokens, num_blocks, num_heads, num_segs, num_frags, seed=0):
    """Build a source/cache pair plus a collision-free slot decomposition."""
    generator = torch.Generator().manual_seed(seed)
    key_scale = torch.randint(
        0,
        256,
        (num_tokens, num_heads, num_frags, VALUES_PER_GROUP),
        dtype=torch.uint8,
        generator=generator,
    )
    cache = torch.zeros(
        (num_blocks, num_heads, num_segs, num_frags, TOKEN_FRAG, VALUES_PER_GROUP),
        dtype=torch.uint8,
    )
    # Distinct (block, seg, frag) per token: advanced-indexing assignment is
    # only well defined when destinations are unique.
    flat = torch.arange(num_tokens, dtype=torch.int64)
    block_ids = flat % num_blocks
    seg_ids = (flat // num_blocks) % num_segs
    frag_ids = (flat // (num_blocks * num_segs)) % TOKEN_FRAG
    return key_scale, cache, block_ids, seg_ids, frag_ids


def _eager_scatter(key_scale, cache, block_ids, seg_ids, frag_ids):
    out = cache.clone()
    out[block_ids, :, seg_ids, :, frag_ids, :] = key_scale
    return out


def _kernel_formula_scatter(key_scale, cache, block_ids, seg_ids, frag_ids):
    """Reproduce the kernel's address arithmetic with torch ops on CPU.

    Mirrors the kernel exactly: bitcast the trailing byte pair away, then
    index the flat storage with
    ``block*st_b + head*st_n + seg*st_s + frag_dim*st_f + frag*st_g``.
    """
    out = cache.clone()
    src = _bitcast_pairs(key_scale)
    dst = _bitcast_pairs(out)
    num_tokens, num_heads, num_frags = src.shape

    token = torch.arange(num_tokens, dtype=torch.int64)[:, None]
    payload = torch.arange(num_heads * num_frags, dtype=torch.int64)[None, :]
    head = payload // num_frags
    frag_dim = payload % num_frags

    src_offset = token * src.stride(0) + head * src.stride(1) + frag_dim * src.stride(2)
    dst_offset = (
        block_ids[:, None] * dst.stride(0)
        + head * dst.stride(1)
        + seg_ids[:, None] * dst.stride(2)
        + frag_dim * dst.stride(3)
        + frag_ids[:, None] * dst.stride(4)
    )

    src_flat = src.reshape(-1)
    dst_flat = dst.reshape(-1)
    dst_flat[dst_offset.reshape(-1)] = src_flat[src_offset.reshape(-1)]
    return out


class TestMxfpKScaleBitcast(TestBase):
    """The trailing ``size=2, stride=1`` byte pair becomes one 16-bit element."""

    def test_bitcast_drops_the_trailing_axis(self):
        tensor = torch.zeros((4, 2, 3, VALUES_PER_GROUP), dtype=torch.uint8)
        self.assertEqual(_bitcast_pairs(tensor).shape, (4, 2, 3))

    def test_bitcast_preserves_byte_order(self):
        tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.uint8)
        viewed = _bitcast_pairs(tensor)
        # Round-tripping must return the original bytes regardless of the
        # host's endianness, which is all the kernel relies on.
        self.assertEqual(viewed.view(torch.uint8).reshape(2, 2).tolist(), [[1, 2], [3, 4]])

    def test_bitcast_rejects_a_non_pair_trailing_axis(self):
        tensor = torch.zeros((4, 3), dtype=torch.uint8)
        with self.assertRaises(ValueError):
            _bitcast_pairs(tensor)

    def test_bitcast_rejects_a_strided_trailing_axis(self):
        tensor = torch.zeros((4, 4), dtype=torch.uint8)[:, ::2]
        with self.assertRaises(ValueError):
            _bitcast_pairs(tensor)

    def test_bitcast_rejects_a_multi_byte_dtype(self):
        tensor = torch.zeros((4, VALUES_PER_GROUP), dtype=torch.int16)
        with self.assertRaises(ValueError):
            _bitcast_pairs(tensor)

    def test_bitcast_rejects_an_odd_page_gap(self):
        # A page stride that is not a multiple of 2 bytes leaves rows off the
        # 16-bit boundary. The trailing pair is still contiguous, so only
        # torch's own view() check catches it; we translate its RuntimeError
        # to ValueError so the caller falls back instead of crashing.
        storage = torch.zeros((3, 5), dtype=torch.uint8)
        odd_gap = torch.as_strided(storage, (3, 2, VALUES_PER_GROUP), (5, 2, 1))
        self.assertEqual(odd_gap.stride(-1), 1)
        with self.assertRaises(ValueError):
            _bitcast_pairs(odd_gap)


class TestMxfpKScaleAddressFormula(TestBase):
    """The kernel's flat addressing must equal the eager 6-D assignment."""

    def test_formula_matches_eager_assignment(self):
        case = _make_case(num_tokens=37, num_blocks=3, num_heads=4, num_segs=2, num_frags=2)
        self.assertTrue(torch.equal(_kernel_formula_scatter(*case), _eager_scatter(*case)))

    def test_formula_matches_for_a_non_power_of_two_fragment_count(self):
        # head_dim 192 -> F = 3, so payload is 12 and payload_p2 is 16: the
        # masked tail lanes must not write anything.
        case = _make_case(num_tokens=20, num_blocks=2, num_heads=4, num_segs=2, num_frags=3, seed=1)
        self.assertTrue(torch.equal(_kernel_formula_scatter(*case), _eager_scatter(*case)))

    def test_formula_matches_for_a_single_head_and_fragment(self):
        case = _make_case(num_tokens=5, num_blocks=2, num_heads=1, num_segs=1, num_frags=1, seed=2)
        self.assertTrue(torch.equal(_kernel_formula_scatter(*case), _eager_scatter(*case)))

    def test_untouched_slots_keep_their_previous_bytes(self):
        key_scale, cache, block_ids, seg_ids, frag_ids = _make_case(
            num_tokens=4, num_blocks=2, num_heads=2, num_segs=2, num_frags=2, seed=3
        )
        cache.fill_(0xAB)
        written = _kernel_formula_scatter(key_scale, cache, block_ids, seg_ids, frag_ids)
        eager = _eager_scatter(key_scale, cache, block_ids, seg_ids, frag_ids)
        self.assertTrue(torch.equal(written, eager))
        # The scatter touches only num_tokens * heads * frags * 2 bytes.
        self.assertEqual(int((written != 0xAB).sum()), key_scale.numel())


class TestMxfpKScaleLaunchGeometry(TestBase):
    """Launch parameters are derived without a Triton runtime or a device."""

    def _launch_kwargs(self, **case_kwargs):
        key_scale, cache, block_ids, seg_ids, frag_ids = _make_case(**case_kwargs)
        target = "vllm_ascend.ops.triton.scatter_mxfp_k_scale._scatter_mxfp_k_scale_kernel"
        with patch(target) as kernel:
            scatter_mxfp_k_scale_cache_triton(key_scale, cache, block_ids, seg_ids, frag_ids)
        grid = kernel.__getitem__.call_args.args[0]
        return grid, kernel.__getitem__.return_value.call_args.kwargs

    def test_payload_is_rounded_up_to_a_power_of_two(self):
        _, kwargs = self._launch_kwargs(num_tokens=8, num_blocks=2, num_heads=4, num_segs=2, num_frags=3)
        self.assertEqual(kwargs["PAYLOAD"], 12)
        self.assertEqual(kwargs["PAYLOAD_P2"], 16)
        self.assertEqual(kwargs["F"], 3)

    def test_grid_covers_every_token(self):
        grid, kwargs = self._launch_kwargs(num_tokens=1000, num_blocks=4, num_heads=4, num_segs=4, num_frags=2)
        block_t = kwargs["BLOCK_T"]
        self.assertGreaterEqual(grid[0] * block_t, 1000)
        self.assertLess((grid[0] - 1) * block_t, 1000)

    def test_tile_stays_within_the_element_budget(self):
        _, kwargs = self._launch_kwargs(num_tokens=64, num_blocks=2, num_heads=64, num_segs=2, num_frags=4)
        self.assertLessEqual(kwargs["BLOCK_T"] * kwargs["PAYLOAD_P2"], 4096)
        self.assertGreaterEqual(kwargs["BLOCK_T"], 1)

    def test_empty_input_launches_nothing(self):
        key_scale, cache, block_ids, seg_ids, frag_ids = _make_case(
            num_tokens=0, num_blocks=2, num_heads=2, num_segs=2, num_frags=2
        )
        target = "vllm_ascend.ops.triton.scatter_mxfp_k_scale._scatter_mxfp_k_scale_kernel"
        with patch(target) as kernel:
            scatter_mxfp_k_scale_cache_triton(key_scale, cache, block_ids, seg_ids, frag_ids)
        kernel.__getitem__.assert_not_called()

    def test_mismatched_cache_axes_are_rejected(self):
        key_scale, cache, block_ids, seg_ids, frag_ids = _make_case(
            num_tokens=4, num_blocks=2, num_heads=2, num_segs=2, num_frags=2
        )
        wrong = cache[:, :1]
        with self.assertRaises(ValueError):
            scatter_mxfp_k_scale_cache_triton(key_scale, wrong, block_ids, seg_ids, frag_ids)
