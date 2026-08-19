#!/usr/bin/env bash
# MassFormer NIST'23 PILOT (fold-swapped) — train -> predict VAL -> eval.
# Train/val = pilot_swapped_scaffold_1.tsv (train on official val fold, validate on
# a molecule-disjoint slice of the official train fold). No test fold; the final
# eval is on the 4,971 pilot VAL specs, selected via the val->test relabelled split.
# Outputs -> results/massformer_baseline_nist23/pilot_rnd1/. GPU 4.
set -euo pipefail
WT=/home/mybirth0407/workspaces/ms-pred/.claude/worktrees/nist23-benchmark
VENV=/home/mybirth0407/workspaces/ms-pred/.venv
export PATH="$VENV/bin:$PATH"
cd "$WT"

GPU=4
EVAL_SPLIT=pilot_swapped_scaffold_1_valeval   # val relabelled as test
RES=results/massformer_baseline_nist23/pilot_rnd1
CKPT=$RES/version_0/best.ckpt

echo "[massformer pilot] TRAIN start $(date '+%F %T')"
python launcher_scripts/run_from_config.py \
  configs/massformer/nist23/massformer_baseline_nist23_pilot.yaml
[ -f "$CKPT" ] || { echo "[massformer pilot] FAIL: no ckpt at $CKPT"; exit 1; }

echo "[massformer pilot] PREDICT VAL + EVAL start $(date '+%F %T')"
SAVE=$RES/preds_val
mkdir -p "$SAVE"
CUDA_VISIBLE_DEVICES=$GPU python src/ms_pred/massformer_pred/predict.py \
  --batch-size 32 --dataset-name nist23 --split-name ${EVAL_SPLIT}.tsv \
  --subset-datasets test_only --checkpoint-pth "$CKPT" \
  --save-dir "$SAVE" --gpu

python analysis/spec_pred_eval.py \
  --binned-pred-file "$SAVE/binned_preds.hdf5" \
  --max-peaks 100 --min-inten 0 \
  --formula-dir-name no_subform.hdf5 --dataset nist23

echo "[massformer pilot] DONE $(date '+%F %T')"
grep -E "avg_cos_sim|avg_entropy_sim|avg_coverage" "$SAVE/pred_eval.yaml" || true
