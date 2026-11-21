# RUNBOOK — Decoder SWA Bounded Replay on Atlas 800 A3

Single entry point. Everything needed is in this directory and in the branch's
code commits; nothing has to be fetched separately.

> **Nothing here has been run.** The operator has not been compiled and no test
> has been executed. Numbers quoted are analytic, not measured. See
> `README.md` for the design and the five open build questions.

---

## 0. Get the code

```bash
git clone https://github.com/chuanbowang2026/vllm-ascend.git
cd vllm-ascend
git checkout feat/decoder-swa-bounded-replay
```

The branch is based on upstream `releases/v0.30.0rc2` (`7d825229`) and adds
six commits. Diff against the base to see only our changes:

```bash
git diff 7d825229..HEAD --stat
```

---

## 1. What to run, in order

Each script prints `PASS` / `FAIL` / `INCONCLUSIVE` on its first line and exits
0 / 1 / 2. **Exit code 2 does not count as a pass.** Run Python with
`python3 -I`.

### Step 1 — no NPU, no weights (run this anywhere, first)

```bash
cd tests/e2e/decoder_swa_bounded_replay

# model structure: layer plan, KV pool, parameter counts
python3 -I probe/03_probe_model.py --config /path/to/DeepSeek-V4.1-Flash/config.json

# mask-vs-compact semantics, exhaustive over chunk boundaries
python3 -I equivalence/l1_visibility_equiv.py
```

Both should PASS before anything else is attempted. `03_probe_model.py`
verifies the premise the whole optimization rests on: that no layer above the
CED boundary owns compressed KV.

### Step 2 — NPU, no weights

```bash
bash probe/01_probe_env.sh          # SOC, CANN, driver, cpuset, devices
python3 -I probe/02_probe_caps.py   # hardware capabilities, compile decorator, cpu binding
```

`02_probe_caps.py` checks two things that change the plan if they come back
differently than expected: that A3 lacks `DSV41_PACKED_CACHE` (so the BF16
cache path is in use), and that the V4.1 model chain carries no
`@support_torch_compile` (so the recipe's `ascend_compilation_config` block is
inert).

### Step 3 — unit tests (no NPU)

```bash
cd <repo root>
pytest -sv tests/ut/models/test_bounded_replay_plan.py
```

### Step 4 — build the operator

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export SOC_VERSION=ascend910_93                        # A3 must match ^ascend910_93
git submodule update --init csrc/third_party/catlass    # pinned commit required
cd csrc && bash build.sh --pkg --ops="sparse_flash_mla;sparse_flash_mla_metadata" --soc="ascend910_93"
./build/cann-ops-transformer*.run --install-path=<repo>/vllm_ascend/_cann_ops_custom
```

**Reinstall the `.run` after every rebuild.** A new `.so` against a stale
installed package fails at runtime with an EZ-series argument-count error, not
a link error — easy to misread as a bug in the patch.

### Step 5 — baseline, unmodified

```bash
bash probe/04_baseline_serve.sh /path/to/weights     # foreground; official recipe, no flags changed
# in another shell, once "Application startup complete" appears:
python3 -I probe/05_bench_prefill.py --url http://127.0.0.1:8900 --model dsv41
```

This is the number everything else is compared against, and it answers the
single most important open question: the **actual MFU**. Every savings
estimate scales with it.

While it runs, these log greps answer three more:

```bash
grep -i 'cpu_bind_mode'                 # did CPU binding apply at all?
grep -i 'enable_npugraph_ex is enabled' # is torch.compile ever reached?
grep -i 'Disabling DSA-CP'              # was DSA-CP silently force-disabled?
```

### Step 6 — single-operator check

```bash
# capture one real consumer-layer call from a running engine
VLLM_ASCEND_DUMP_SMLA=/tmp/smla.pt python3 -I -c \
  "import capture_smla_args; capture_smla_args.install(); \
   from vllm.entrypoints.cli.main import main; main()" serve <MODEL> ...

python3 -I test_ori_win_floor.py --dump /tmp/smla.pt
```

**K4 is the check that matters.** K1–K3 (absent floor unchanged, zero floor a
no-op, out-of-range floor safe) all pass for a patch that silently ignores the
new operator input. Only K4 — an in-range floor must change the output —
catches that.

### Step 7 — end to end

```bash
# baseline on :8900, mask mode on :8901
python3 -I test_e2e_mask_mode.py --baseline http://127.0.0.1:8900 \
                                  --candidate http://127.0.0.1:8901
```

**Expect flat throughput.** `mask` narrows visibility but drops no rows, so it
saves no compute by design. A large regression means the patch added overhead;
it does not mean the optimization underperformed. The speedup comes from
`compact`, which is not implemented on this branch.

---

## 2. Turning it on

```
--additional-config '{ ...,
    "enable_decoder_swa_bounded_replay": true,
    "decoder_swa_replay_mode": "mask" }'
```

`decoder_swa_replay_keep_tokens` defaults to `sliding_window + num_spec + 1`
= **129** with no speculative decoding. Setting it lower raises.

Refused outright (none silently degrade): prefix caching, the vision path,
the packed-cache path, and `decoder_swa_replay_mode="compact"`.

---

## 3. What to send back

```
tests/e2e/decoder_swa_bounded_replay/*/results/*.json
```

Plus the serve logs. The three numbers that decide what gets built next:

| | why it matters |
|---|---|
| actual MFU | every savings estimate scales with it; if MFU is far below ~25 %, compute is not the binding constraint and the EP-communication work outranks the layer skip |
| EP all-to-all measured time | reconcile against 7.55 GB/rank/step. Near 22 ms means it is not co-dominant and byte-reduction work drops in priority; far above means the bottleneck is barriers or imbalance, not bytes |
| expert load rank max/mean | sampling floor is 1.016. Measured 1.05 means EPLB's ceiling is ~5 % and is not worth it; 1.4 means it is the second lever |

---

## 4. One thing worth measuring early, before `compact` is written

After the CED boundary layer, each rank would hold `ceil(total_keep / tp)`
rows instead of `ceil(num_tokens / tp)`. If any DP or MC2 collective is sized
from a token count negotiated **before** the forward pass, it will disagree.

SGLang needed dedicated machinery for exactly this
(`enter_late_layer_dp` / `late_layer_dp_counts`).

Printing the per-rank row count at the boundary layer on the current build
answers it without writing any of `compact`, and the answer may change how
`compact` has to be built. It is the single highest-value measurement on this
list after MFU.
