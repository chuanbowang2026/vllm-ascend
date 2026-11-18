# SPDX-License-Identifier: Apache-2.0
"""CED boundary derivation for Decoder SWA Bounded Replay.

Runs on CPU via the repo's mocked torch_npu (tests/ut/conftest.py), so this
needs no NPU. Copy into tests/ut/models/ and run:

    pytest -sv tests/ut/models/test_bounded_replay_plan.py
"""

from types import SimpleNamespace

import pytest

from vllm_ascend.models.deepseek_v41.model import build_layer_plan

# The released DeepSeek-V4.1-Flash topology (checkpoint config.json).
RELEASED = dict(
    num_hidden_layers=40,
    compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    candidate_source_layer_id=20,
    candidate_topk_blocks=2048,
    candidate_block_size=8,
    index_topk=512,
    engram_layer_ids=[1, 14],
)


def _config(**overrides):
    return SimpleNamespace(**{**RELEASED, **overrides})


def test_boundary_is_last_kv_source():
    topo = build_layer_plan(_config())
    assert topo.boundary_layer_idx == 20


def test_boundary_splits_the_stack_in_half():
    topo = build_layer_plan(_config())
    consumers = [r.layer_idx for r in topo.layers if r.is_swa_replay_consumer]
    producers = [r.layer_idx for r in topo.layers if not r.is_swa_replay_consumer]
    assert producers == list(range(0, 20))
    assert consumers == list(range(20, 40))


def test_no_layer_above_the_boundary_owns_compressed_kv():
    """The premise of the whole optimization.

    Prefill may stop computing layers above the boundary for most tokens only
    because none of them projects its own global KV -- they all read the
    boundary layer's. If this ever fails, bounded replay is invalid.
    """
    topo = build_layer_plan(_config())
    owners = [r.layer_idx for r in topo.layers if r.is_kv_source and r.layer_idx > topo.boundary_layer_idx]
    assert owners == []
    for role in topo.layers:
        if role.layer_idx > topo.boundary_layer_idx:
            assert role.kv_source_layer == topo.boundary_layer_idx


def test_index_consumers_above_boundary_are_not_sources():
    """Layers 24/28/32/36 select from the boundary layer's cache, own nothing."""
    topo = build_layer_plan(_config())
    for layer_idx in (24, 28, 32, 36):
        role = topo.layer(layer_idx)
        assert role.is_index_source is True
        assert role.is_kv_source is False
        assert role.uses_candidate_filter is True


def test_compressor_ring_only_below_boundary():
    topo = build_layer_plan(_config())
    rings = [r.layer_idx for r in topo.layers if r.is_kv_source and r.compress_ratio == 2]
    assert rings == [2, 8, 14]
    assert all(idx < topo.boundary_layer_idx for idx in rings)


def test_engram_layers_are_below_boundary():
    """Engram in a consumer layer would break the replay; assert it is not."""
    topo = build_layer_plan(_config())
    engram = [r.layer_idx for r in topo.layers if r.engram_slot is not None]
    assert engram == [1, 14]
    assert all(idx < topo.boundary_layer_idx for idx in engram)


def test_ambiguous_topology_is_rejected_not_guessed():
    """Two independent derivations of the boundary must agree.

    max(kv_source_layer_ids) and the first compress_ratio==1 layer coincide by
    constraint, not by definition. A checkpoint where they disagree is not the
    CED shape this plan assumes, so refuse rather than silently pick one.
    """
    bad = _config(compress_ratios=[0, 0] + [2] * 16 + [1] * 22 + [0, 0, 0])
    with pytest.raises(ValueError, match="CED boundary is ambiguous"):
        build_layer_plan(bad)


def test_role_default_keeps_non_ced_models_working():
    """A model with no kv sources must not be marked as having a boundary."""
    topo = build_layer_plan(
        _config(kv_source_layer_ids=[], index_source_layer_ids=[], compress_ratios=[0] * 40)
    )
    assert topo.boundary_layer_idx == -1
    assert not any(r.is_swa_replay_consumer for r in topo.layers)
