#!/usr/bin/env bash
# Stage-0 orchestrator. Runs every probe whose preconditions are met and
# prints a summary table. Does NOT start the service (that is interactive) —
# it will use an already-running one if --url is reachable.
#
# Usage:
#   bash run_all.sh /path/to/DeepSeek-V4.1-Flash-w8a8 [http://127.0.0.1:8900]
#
# Exit code: 0 if every probe PASSed, 1 if any FAILed, 2 if any was INCONCLUSIVE
# and none FAILed.  (2 does NOT count as a pass.)

set -uo pipefail
cd "$(dirname "$0")"

MODEL_PATH="${1:-}"
URL="${2:-http://127.0.0.1:8900}"
export PROBE_RESULTS_DIR="${PROBE_RESULTS_DIR:-results}"
mkdir -p "$PROBE_RESULTS_DIR"

declare -a NAMES=() CODES=()

record() { NAMES+=("$1"); CODES+=("$2"); }

banner() { echo; echo "================================================================"; echo "== $*"; echo "================================================================"; }

# --- 03: pure CPU, needs only config.json -----------------------------------
banner "03_probe_model (no NPU, no weights — only config.json)"
CFG=""
if [ -n "$MODEL_PATH" ] && [ -f "$MODEL_PATH/config.json" ]; then
  CFG="$MODEL_PATH/config.json"
elif [ -f "./config.json" ]; then
  CFG="./config.json"
fi
if [ -n "$CFG" ]; then
  python3 -I 03_probe_model.py --config "$CFG"; record 03_probe_model $?
else
  echo "INCONCLUSIVE"; echo "# no config.json found (pass the model dir as \$1)"
  record 03_probe_model 2
fi

# --- 01: needs NPU ----------------------------------------------------------
banner "01_probe_env (needs NPU visible)"
bash 01_probe_env.sh; record 01_probe_env $?

# --- 02: needs NPU + vllm_ascend -------------------------------------------
banner "02_probe_caps (needs NPU + vllm_ascend importable)"
python3 -I 02_probe_caps.py; record 02_probe_caps $?

# --- 05: needs a running service -------------------------------------------
banner "05_bench_prefill (needs a running service at $URL)"
if curl -sS -o /dev/null -m 10 "$URL/v1/models" 2>/dev/null; then
  python3 -I 05_bench_prefill.py --url "$URL" --model dsv41; record 05_bench_prefill $?
else
  echo "INCONCLUSIVE"
  echo "# no service at $URL — start it first:"
  echo "#   bash 04_baseline_serve.sh ${MODEL_PATH:-<MODEL_PATH>}"
  record 05_bench_prefill 2
fi

# --- summary ----------------------------------------------------------------
banner "SUMMARY"
WORST=0
for i in "${!NAMES[@]}"; do
  c="${CODES[$i]}"
  case "$c" in 0) v=PASS ;; 1) v=FAIL ;; *) v=INCONCLUSIVE ;; esac
  printf '  %-20s %s\n' "${NAMES[$i]}" "$v"
  if [ "$c" -eq 1 ]; then WORST=1; elif [ "$c" -ne 0 ] && [ "$WORST" -ne 1 ]; then WORST=2; fi
done
echo
echo "results written to: $PROBE_RESULTS_DIR/"
echo "send that whole directory back for analysis."
case "$WORST" in 0) echo "overall: PASS" ;; 1) echo "overall: FAIL" ;; *) echo "overall: INCONCLUSIVE" ;; esac
exit "$WORST"
