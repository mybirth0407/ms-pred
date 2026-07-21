#!/usr/bin/env bash
# GLACIER NIST'23 — add spectral intensity to the MAGMa heuristic tree.
# Produces magma_outputs/magma_tree_with_inten.hdf5, which train_joint.py requires.
# Depends on 00_preprocess.sh having produced magma_outputs/magma_tree.hdf5 and
# subformulae/no_subform.hdf5.
set -euo pipefail

dataset="nist23"
PRED_MAGMA_H5="data/spec_datasets/$dataset/magma_outputs/magma_tree.hdf5"
TRUE_DAG_H5="data/spec_datasets/$dataset/subformulae/no_subform.hdf5"
OUT_MAGMA_H5="data/spec_datasets/$dataset/magma_outputs/magma_tree_with_inten.hdf5"

python data_scripts/dag/add_dag_intens.py \
  --pred-dag-path "$PRED_MAGMA_H5" \
  --true-dag-path "$TRUE_DAG_H5" \
  --out-dag-path "$OUT_MAGMA_H5" \
  --num-workers 32 \
  --magma-output
