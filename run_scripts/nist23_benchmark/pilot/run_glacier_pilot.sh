#!/usr/bin/env bash
# GLACIER NIST'23 PILOT (fold-swapped) — joint train (single-GPU) -> predict VAL -> eval.
# Train/val = pilot_swapped_scaffold_1.tsv. Final eval on the 4,971 pilot VAL specs via
# the val->test relabelled split. Outputs -> results/glacier_nist23/pilot_rnd1/. GPU 5.
set -euo pipefail
WT=/home/mybirth0407/workspaces/ms-pred/.claude/worktrees/nist23-benchmark
VENV=/home/mybirth0407/workspaces/ms-pred/.venv
export PATH="$VENV/bin:$PATH"
cd "$WT"

GPU=5
EVAL_SPLIT=pilot_swapped_scaffold_1_valeval
RES=results/glacier_nist23/pilot_rnd1
CKPT=$RES/version_0/best.ckpt

echo "[glacier pilot] TRAIN start $(date '+%F %T')"
python launcher_scripts/run_from_config.py \
  configs/glacier/nist23/joint_train_nist23_pilot.yaml
[ -f "$CKPT" ] || { echo "[glacier pilot] FAIL: no ckpt at $CKPT"; exit 1; }

echo "[glacier pilot] PREDICT VAL + EVAL start $(date '+%F %T')"
SAVE=$RES/preds_val
mkdir -p "$SAVE"
CUDA_VISIBLE_DEVICES=$GPU python src/ms_pred/glacier/predict_inten_joint.py \
  --batch-size 64 --dataset-name nist23 --split-name ${EVAL_SPLIT}.tsv \
  --checkpoint-pth "$CKPT" --save-dir "$SAVE" --gpu --num-workers 64 \
  --subset-datasets test_only --binned-out

python analysis/spec_pred_eval.py \
  --binned-pred-file "$SAVE/binned_preds.hdf5" \
  --max-peaks 100 --min-inten 0 \
  --formula-dir-name no_subform.hdf5 --dataset nist23

echo "[glacier pilot] DONE $(date '+%F %T')"
grep -E "avg_cos_sim|avg_entropy_sim|avg_coverage" "$SAVE/pred_eval.yaml" || true
