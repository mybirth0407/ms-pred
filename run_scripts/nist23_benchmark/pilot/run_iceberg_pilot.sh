#!/usr/bin/env bash
# ICEBERG NIST'23 PILOT (fold-swapped) — full 2-stage pipeline -> predict VAL -> eval.
#   1. gen train         (GPU 6)   -> ckpt/gen/best.ckpt
#   2. gen self-predict   (GPU 6,7) -> preds_train_100/tree_preds.hdf5 (all pilot specs)
#   3. add_dag_intens     (CPU)    -> preds_train_100_inten.hdf5
#   4. inten train        (GPU 6)   -> ckpt/inten/best.ckpt
#   5. predict VAL + eval (GPU 6)   -> preds_val/ (4,971 pilot val specs)
# Train/val = pilot_swapped_scaffold_1.tsv; eval selects the val fold via the
# val->test relabelled split. Outputs -> results/iceberg_nist23/pilot_rnd1/.
set -euo pipefail
WT=/home/mybirth0407/workspaces/ms-pred/.claude/worktrees/nist23-benchmark
VENV=/home/mybirth0407/workspaces/ms-pred/.venv
export PATH="$VENV/bin:$PATH"
cd "$WT"

EVAL_SPLIT=pilot_swapped_scaffold_1_valeval
RES=results/iceberg_nist23/pilot_rnd1
CFG=configs/iceberg/nist23

# Each stage is guarded on its output so a re-run resumes instead of repeating the
# ~2h gen training (or any completed stage).
if [ -f "$RES/ckpt/gen/best.ckpt" ]; then
  echo "[iceberg pilot] 1/5 GEN TRAIN skip (ckpt exists) $(date '+%F %T')"
else
  echo "[iceberg pilot] 1/5 GEN TRAIN start $(date '+%F %T')"
  python launcher_scripts/run_from_config.py $CFG/dag_train_nist23_pilot.yaml
  [ -f "$RES/ckpt/gen/best.ckpt" ] || { echo "[iceberg pilot] FAIL: no gen ckpt"; exit 1; }
fi

if [ -f "$RES/preds_train_100/tree_preds.hdf5" ]; then
  echo "[iceberg pilot] 2/5 GEN SELF-PREDICT skip (exists) $(date '+%F %T')"
else
  echo "[iceberg pilot] 2/5 GEN SELF-PREDICT start $(date '+%F %T')"
  python launcher_scripts/run_from_config.py $CFG/dag_gen_predict_train_nist23_pilot.yaml
  [ -f "$RES/preds_train_100/tree_preds.hdf5" ] || { echo "[iceberg pilot] FAIL: no tree_preds"; exit 1; }
fi

if [ -f "$RES/preds_train_100_inten.hdf5" ]; then
  echo "[iceberg pilot] 3/5 ADD DAG INTENS skip (exists) $(date '+%F %T')"
else
  echo "[iceberg pilot] 3/5 ADD DAG INTENS start $(date '+%F %T')"
  python data_scripts/dag/add_dag_intens.py \
    --pred-dag-path  "$RES/preds_train_100/tree_preds.hdf5" \
    --true-dag-path  data/spec_datasets/nist23/subformulae/no_subform.hdf5 \
    --out-dag-path   "$RES/preds_train_100_inten.hdf5" \
    --num-workers 32
  [ -f "$RES/preds_train_100_inten.hdf5" ] || { echo "[iceberg pilot] FAIL: no inten dag"; exit 1; }
fi

if [ -f "$RES/ckpt/inten/best.ckpt" ]; then
  echo "[iceberg pilot] 4/5 INTEN TRAIN skip (ckpt exists) $(date '+%F %T')"
else
  echo "[iceberg pilot] 4/5 INTEN TRAIN start $(date '+%F %T')"
  python launcher_scripts/run_from_config.py $CFG/dag_inten_train_nist23_pilot.yaml
  [ -f "$RES/ckpt/inten/best.ckpt" ] || { echo "[iceberg pilot] FAIL: no inten ckpt"; exit 1; }
fi

echo "[iceberg pilot] 5/5 PREDICT VAL + EVAL start $(date '+%F %T')"
SAVE=$RES/preds_val
mkdir -p "$SAVE"
CUDA_VISIBLE_DEVICES=6 python src/ms_pred/iceberg/predict_inten.py \
  --batch-size 64 --dataset-name nist23 --split-name ${EVAL_SPLIT}.tsv \
  --checkpoint-pth "$RES/ckpt/inten/best.ckpt" \
  --save-dir "$SAVE" --gpu --num-workers 0 \
  --magma-dag-folder "$RES/preds_train_100/tree_preds.hdf5" \
  --subset-datasets test_only --binned-out

python analysis/spec_pred_eval.py \
  --binned-pred-file "$SAVE/binned_preds.hdf5" \
  --max-peaks 100 --min-inten 0 \
  --formula-dir-name no_subform.hdf5 --dataset nist23

echo "[iceberg pilot] DONE $(date '+%F %T')"
grep -E "avg_cos_sim|avg_entropy_sim|avg_coverage" "$SAVE/pred_eval.yaml" || true
