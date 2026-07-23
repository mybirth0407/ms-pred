#!/usr/bin/env bash
# =============================================================================
# Stage-1 fragment-DAG quality benchmark for the 2-stage fragment models
# (ICEBERG and MARASON) on the NIST'23 scaffold_1 split.
#
# For each model this script (optionally) runs the generative stage on the TEST
# split to emit a predicted fragment DAG (tree_preds.hdf5), then scores it
# against the gold MAGMa DAG (magma_tree.hdf5) with analysis/dag_frag_eval.py,
# and finally prints a combined PRECISION / RECALL / F1 / JACCARD table.
#
# SAFE BY DEFAULT: the GPU prediction step does NOT run unless you pass
# RUN_PREDICT=1. With RUN_PREDICT unset the script only reads existing
# tree_preds.hdf5 files and computes metrics on CPU. This lets it coexist with
# other jobs training on the machine.
#
# ---- usage ------------------------------------------------------------------
#   # CPU-only: score already-generated test predictions and print the table
#   bash run_scripts/nist23_benchmark/stage1_frag_eval.sh
#
#   # Full run: generate TEST-split DAGs on GPU 0, then score them
#   RUN_PREDICT=1 GPU=0 bash run_scripts/nist23_benchmark/stage1_frag_eval.sh
#
# ---- env knobs (all optional) -----------------------------------------------
#   RUN_PREDICT    1 = run predict_gen.py on GPU first; unset/0 = eval only   [0]
#   GPU            CUDA device index for the prediction step                  [0]
#   MS_PRED_ROOT   canonical checkout (holds .venv + editable src + data)
#                                          [/home/mybirth0407/workspaces/ms-pred]
#   PY             python interpreter                        [$MS_PRED_ROOT/.venv/bin/python]
#   DATASET        dataset name                                          [nist23]
#   SPLIT          split file under data/spec_datasets/$DATASET/splits [scaffold_1.tsv]
#   SPLIT_TAG      results sub-directory tag                      [scaffold_1_rnd1]
#   PREDS_SUBDIR   prediction output dir name                      [preds_test_100]
#   IDENTITY       fragment identity: wl | formula | atoms                   [wl]
#   THRESHOLD      predict_gen node-prob threshold                          [0.0]
#   MAX_NODES      predict_gen max nodes per DAG                            [100]
#   EVAL_MAX_CPU   parallel workers for the CPU eval                        [16]
#   MODELS         space-separated subset of {iceberg marason}   [iceberg marason]
# =============================================================================
set -euo pipefail

MS_PRED_ROOT="${MS_PRED_ROOT:-/home/mybirth0407/workspaces/ms-pred}"
PY="${PY:-${MS_PRED_ROOT}/.venv/bin/python}"
GPU="${GPU:-0}"
RUN_PREDICT="${RUN_PREDICT:-0}"

DATASET="${DATASET:-nist23}"
SPLIT="${SPLIT:-scaffold_1.tsv}"
SPLIT_TAG="${SPLIT_TAG:-scaffold_1_rnd1}"
PREDS_SUBDIR="${PREDS_SUBDIR:-preds_test_100}"
IDENTITY="${IDENTITY:-wl}"
THRESHOLD="${THRESHOLD:-0.0}"
MAX_NODES="${MAX_NODES:-100}"
EVAL_MAX_CPU="${EVAL_MAX_CPU:-16}"
MODELS="${MODELS:-iceberg marason}"

# dag_frag_eval.py lives with THIS script's worktree (may not yet be in the
# canonical checkout), so resolve it by the script's own location.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="$(cd "${SCRIPT_DIR}/../.." && pwd)/analysis/dag_frag_eval.py"

GOLD_TREE="data/spec_datasets/${DATASET}/magma_outputs/magma_tree.hdf5"

# Everything is repo-relative to the canonical checkout (import ms_pred + data).
cd "${MS_PRED_ROOT}"

echo "=========================================================================="
echo " NIST'23 stage-1 fragment-DAG eval | dataset=${DATASET} split=${SPLIT}"
echo " identity=${IDENTITY}  RUN_PREDICT=${RUN_PREDICT}  models=${MODELS}"
echo " eval script: ${EVAL_SCRIPT}"
echo "=========================================================================="

# per-model config: predict_gen script, gen checkpoint, batch size
model_gen_script() {  # $1 = model
  case "$1" in
    iceberg) echo "src/ms_pred/iceberg/predict_gen.py" ;;
    marason) echo "src/ms_pred/marason/predict_gen.py" ;;
    *) echo "UNKNOWN" ;;
  esac
}
model_batch_size() { case "$1" in iceberg) echo 96 ;; marason) echo 128 ;; *) echo 64 ;; esac; }
model_results_prefix() { echo "results/${1}_${DATASET}"; }   # e.g. results/iceberg_nist23

EVAL_YAMLS=()   # collect written yaml paths for the combined table
EVAL_MODELS=()

for MODEL in ${MODELS}; do
  RES_PREFIX="$(model_results_prefix "${MODEL}")"
  CKPT="${RES_PREFIX}/${SPLIT_TAG}/ckpt/gen/best.ckpt"
  SAVE_DIR="${RES_PREFIX}/${SPLIT_TAG}/${PREDS_SUBDIR}"
  PRED_TREE="${SAVE_DIR}/tree_preds.hdf5"
  EVAL_YAML="${SAVE_DIR}/pred_eval_frag.yaml"
  GEN_SCRIPT="$(model_gen_script "${MODEL}")"

  echo
  echo "----- ${MODEL} -----------------------------------------------------------"
  echo "  checkpoint : ${CKPT}"
  echo "  save-dir   : ${SAVE_DIR}"

  # -------- STEP 1: generate TEST-split DAGs (GPU, guarded) -----------------
  if [[ "${RUN_PREDICT}" == "1" ]]; then
    echo "  [predict] RUN_PREDICT=1 -> generating test DAGs on GPU ${GPU}"
    CUDA_VISIBLE_DEVICES="${GPU}" "${PY}" "${GEN_SCRIPT}" \
      --gpu \
      --dataset-name "${DATASET}" \
      --dataset-labels labels.tsv \
      --split-name "${SPLIT}" \
      --subset-datasets test_only \
      --checkpoint-pth "${CKPT}" \
      --save-dir "${SAVE_DIR}" \
      --batch-size "$(model_batch_size "${MODEL}")" \
      --num-gpu-workers 0 \
      --num-cpu-workers 16 \
      --threshold "${THRESHOLD}" \
      --max-nodes "${MAX_NODES}" \
      --num-decoys 0
  else
    echo "  [predict] skipped (RUN_PREDICT!=1). Expecting existing ${PRED_TREE}"
    # --- template of the exact command RUN_PREDICT=1 would execute ---
    #   CUDA_VISIBLE_DEVICES=${GPU} ${PY} ${GEN_SCRIPT} \
    #     --gpu --dataset-name ${DATASET} --dataset-labels labels.tsv \
    #     --split-name ${SPLIT} --subset-datasets test_only \
    #     --checkpoint-pth ${CKPT} --save-dir ${SAVE_DIR} \
    #     --batch-size $(model_batch_size ${MODEL}) --num-gpu-workers 0 \
    #     --num-cpu-workers 16 --threshold ${THRESHOLD} --max-nodes ${MAX_NODES} \
    #     --num-decoys 0
  fi

  # -------- STEP 2: score against gold (CPU, read-only) --------------------
  if [[ ! -f "${PRED_TREE}" ]]; then
    echo "  [eval] SKIP ${MODEL}: ${PRED_TREE} not found (run with RUN_PREDICT=1 first)."
    continue
  fi
  echo "  [eval] scoring ${PRED_TREE}"
  "${PY}" "${EVAL_SCRIPT}" \
    --pred-tree-h5 "${PRED_TREE}" \
    --gold-tree-h5 "${GOLD_TREE}" \
    --dataset "${DATASET}" \
    --identity "${IDENTITY}" \
    --max-cpu "${EVAL_MAX_CPU}" \
    --outfile "${EVAL_YAML}"

  EVAL_YAMLS+=("${EVAL_YAML}")
  EVAL_MODELS+=("${MODEL}")
done

# -------- STEP 3: combined table ------------------------------------------
if [[ "${#EVAL_YAMLS[@]}" -eq 0 ]]; then
  echo
  echo "No evaluations produced (no tree_preds.hdf5 found). Re-run with RUN_PREDICT=1."
  exit 0
fi

echo
echo "=========================================================================="
echo " COMBINED STAGE-1 FRAGMENT-DAG TABLE (identity=${IDENTITY})"
echo "=========================================================================="
MODELS_CSV="$(IFS=,; echo "${EVAL_MODELS[*]}")"
YAMLS_CSV="$(IFS=,; echo "${EVAL_YAMLS[*]}")"
COMBINED_TSV="results/nist23_stage1_frag_${SPLIT_TAG}_${IDENTITY}.tsv"

MODELS_CSV="${MODELS_CSV}" YAMLS_CSV="${YAMLS_CSV}" COMBINED_TSV="${COMBINED_TSV}" "${PY}" - <<'PYEOF'
import os, yaml
models = os.environ["MODELS_CSV"].split(",")
yamls = os.environ["YAMLS_CSV"].split(",")
out_tsv = os.environ["COMBINED_TSV"]

cols = ["precision", "recall", "f1", "jaccard"]
header = f"{'model':<10} {'n_entries':>9} " + " ".join(f"{c:>18}" for c in cols) + \
         f" {'avg_num_true':>12} {'avg_num_pred':>12}"
print(header)
print("-" * len(header))
rows = []
for m, y in zip(models, yamls):
    d = yaml.safe_load(open(y))
    cells = []
    for c in cols:
        cells.append(f"{d[f'avg_{c}']:.4f}+/-{d[f'sem_{c}']:.4f}")
    line = f"{m:<10} {d['num_matched_entries']:>9} " + " ".join(f"{c:>18}" for c in cells) + \
           f" {d['avg_num_true']:>12.2f} {d['avg_num_pred']:>12.2f}"
    print(line)
    rows.append({"model": m, "n_entries": d["num_matched_entries"],
                 **{f"avg_{c}": d[f"avg_{c}"] for c in cols},
                 **{f"sem_{c}": d[f"sem_{c}"] for c in cols},
                 "avg_num_true": d["avg_num_true"], "avg_num_pred": d["avg_num_pred"]})

# also write a machine-readable TSV
import csv
with open(out_tsv, "w", newline="") as fp:
    w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()), delimiter="\t")
    w.writeheader()
    w.writerows(rows)
print(f"\nwrote combined table: {out_tsv}")
PYEOF

echo
echo "Done."
