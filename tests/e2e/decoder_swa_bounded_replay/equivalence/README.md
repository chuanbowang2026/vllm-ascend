# Stage 2 — mask vs compact equivalence

> The only way to know whether `compact` is correct.

## Why this exists before compact ships

`compact` drops query rows. `mask` keeps them and only narrows what they can
see. If compact is right, the rows it keeps performed **exactly** the
computation mask's corresponding rows performed — same weights, same inputs,
same visible keys — so the two must agree **bit for bit**.

That makes mask a usable oracle and makes the comparison exact. There is no
`rtol`, no `atol`, no similarity score anywhere in this directory.

**`off` is deliberately never the reference.** Comparing compact against full
prefill would measure the *approximation* — how much the bounded replay changes
the model's answer. That is a model-quality question which DeepSeek settled in
post-training (the paper's "train-aware adaptation"), not an implementation
question. Mixing the two would mean a real bug could hide behind "well, it's
approximate".

DeepInfer enforces the same direction in its own harness:

```python
assert ref_impls == {IMPL_MASK} and cand_impls == {IMPL_COMPACTION}
```

## Two levels

| | needs | catches |
|---|---|---|
| **L1** `l1_visibility_equiv.py` | nothing — pure Python, < 1 s | off-by-ones in the row selection and the visibility interval; exhaustive chunk-boundary coverage |
| **L2** `l2_engine_equiv.py` | 2 services + weights | wrong gather indices, inconsistently compacted metadata, stale shared buffers — anything numeric |

They are complementary, and L1 covers cases L2 structurally cannot: at L2 the
scheduler picks the chunk boundaries, so you cannot ask it for "a chunk that
ends exactly one token before the rectangle".

### L1

```bash
python3 -I l1_visibility_equiv.py
python3 -I l1_visibility_equiv.py --window 128 --keep 134 --verbose
```

Enumerates prompts × chunk sizes across every boundary neighbourhood
(1, 2, W−1, W, W+1, 255/256/257, 8191/8192/8193, 150 000) and asserts five
properties per case:

1. every row compact keeps agrees with mask **exactly**
2. rows mask keeps but compact drops are all below the rectangle — the one
   sanctioned difference
3. the surviving set matches a closed-form reference, except for the mandatory
   placeholder row when a whole chunk falls below the floor
4. **the safety property**: a surviving row never reads a key below the
   rectangle, because compact never wrote those. DeepInfer asserts the same
   thing in-engine (`TORCH_CHECK_GE(vis.k_lo, clip_prefix)`)
5. every chunk keeps ≥ 1 row, so the sampler always has an addressable output

**This harness was mutation-tested.** Four injected bugs — an off-by-one in
`clip_prefix`, anchoring the floor on the chunk instead of the full prompt
(SGLang's convention leaking into DeepInfer's), dropping the floor from the
visibility computation, and keeping one row too many — are all caught. A test
that cannot fail is worth nothing, so this is worth re-running after any edit
to the semantics.

### L2

```bash
bash run_both_modes.sh /path/to/weights        # launches both, one flag apart
# wait for "Application startup complete" in both logs
python3 -I l2_engine_equiv.py --mask http://127.0.0.1:8900 \
                              --compact http://127.0.0.1:8901
```

Greedy, fixed seed, `logprobs=5`, so the comparison sees numbers rather than
just text. Compares generated token ids, per-token logprobs and top-k
logprobs with `==`.

Prompt lengths are chosen to straddle the rectangle: inside it, just past it,
and long enough that early chunks fall entirely below the floor and keep only
the placeholder row.

**`run_both_modes.sh` launches both services from one script with exactly one
flag different.** That is not convenience — the harness cannot verify the two
configs match, so any other difference (seed, batch limits, parallel layout,
a `VLLM_ASCEND_*` variable) appears as a false failure. It runs them DP2/TP4
side by side rather than the production DP4/TP4, because both must be resident
at once; the modes only have to be identical to *each other*.

## Reading a failure

A mismatch is a **bug in compact**, not drift. Likely causes, in the order
worth checking:

1. a wrong gather index at the boundary layer
2. metadata compacted inconsistently across the 10 SWA cache groups — note
   every group straddles the boundary (`cache_config.py:166`,
   `target_swa[slot_idx::len(full)]`), so a group-level override is not enough
3. a shared index buffer (`topk_indices_buffer`, `candidate_*`) not gathered
   when the stream compacts at layer 20

L2 reports the first divergence per case — the token index and the two values
— so you can tell "diverged at the first generated token" (metadata or gather)
from "diverged at token 20" (something stateful in decode).

## What neither level covers

- The approximation itself. By construction. Use the stage-1 e2e script
  against `off` if you want that number, and read it as a model-quality
  measurement, not a correctness one.
- Performance. Stage 3's benchmark does that; compact is where throughput
  should finally move.
- Prefix caching and the vision path — both refused in config for now.
