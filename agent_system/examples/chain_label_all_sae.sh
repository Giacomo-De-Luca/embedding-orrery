#!/bin/bash
# Serial label/eval/score chain for all 6 SAE max_prefill configs
# (3 PT + 3 IT). Idempotent and rate-limit-resumable:
#
# - Each config is invoked via scripts/resume_label_config.py, which inspects
#   the per-store state (labels/, evaluator/, scores.parquet) and:
#     * does nothing if scores.parquet exists
#     * skips already-finished stages
#     * pre-seeds the global results dir with any saved per-store features,
#       so mid-stage rate-limit kills only re-run the missing features.
# - Configs are serial because both PT and IT SAE runs share the same global
#   queue (autointerpret-{label,eval}).
#
# Launch (sandbox-disabled):
#   nohup caffeinate -is bash scripts/chain_label_all_sae.sh \
#     >> resources/sae_autointerpret/_logs/chain_label_all_sae.log 2>&1 & disown
#
# Manually resume after rate-limit kill:
#   just re-launch with the same command above. The chain skips finished
#   configs and finished stages automatically.

set -u
cd /Users/jack/Colour_vectors || exit 1

LOGDIR=resources/sae_autointerpret/_logs
mkdir -p "$LOGDIR"

CFGS=(
  # PT L9/L29 w16k already scored from previous runs (no-op skip on resume).
  interpret/configs/autointerpret/label_gemma_pt_L9_w16k_max_200_sonnet.yaml
  interpret/configs/autointerpret/label_gemma_pt_L29_w16k_max_200_sonnet.yaml
  # IT runs first — smaller w16k extracts, less likely to OOM than w65k.
  interpret/configs/autointerpret/label_gemma_it_L9_w16k_max_200_sonnet.yaml
  interpret/configs/autointerpret/label_gemma_it_L29_w16k_max_200_sonnet.yaml
  # w65k runs at the end — they keep killing during extract (OOM on the
  # 212k x 65536 sparse matrix load). If they OOM here, the IT w16k runs
  # are already done; only the w65k pair is left blocked.
  interpret/configs/autointerpret/label_gemma_it_L29_w65k_max_200_sonnet.yaml
  interpret/configs/autointerpret/label_gemma_pt_L29_w65k_max_200_sonnet.yaml
)

# Skip-on-error: if any config returns non-zero (rate-limit OR OOM), the chain
# previously exit-aborted. Switch to skip-and-continue so the failing config
# (likely w65k) doesn't block the rest. Manually relaunch the chain to retry
# any skipped config.
SKIP_ON_ERROR=true

# PID-file guard so a second invocation no-ops instead of double-launching.
LOCK="$LOGDIR/chain_label_all_sae.pid"
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
  echo "[chain $(date)] another instance (pid $(cat "$LOCK")) already running, exiting"
  exit 0
fi
echo $$ > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

for cfg in "${CFGS[@]}"; do
  echo "[chain $(date)] === START $cfg ==="
  uv run python scripts/resume_label_config.py "$cfg"
  rc=$?
  echo "[chain $(date)] === END $cfg (exit $rc) ==="
  if [ $rc -ne 0 ]; then
    if $SKIP_ON_ERROR; then
      echo "[chain $(date)] skipping $cfg (rc=$rc) and continuing — re-launch chain to retry"
      continue
    fi
    echo "[chain $(date)] aborting chain on non-zero exit (probably rate-limit; re-launch chain to resume)"
    exit $rc
  fi
done

echo "[chain $(date)] all SAE label runs attempted"
