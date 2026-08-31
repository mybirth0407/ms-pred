#!/usr/bin/env bash
# MARASON NIST'23 — stage 2: predict fragment DAGs for the training molecules and
# attach intensities so stage 3 (intensity model) can train.
set -euo pipefail

python launcher_scripts/run_from_config.py configs/marason/nist23/marason_gen_predict_train_nist23.yaml

for folder in scaffold_1_rnd1; do
  python data_scripts/dag/add_dag_intens.py \
    --pred-dag-path results/marason_nist23/${folder}/preds_train_100/tree_preds.hdf5 \
    --true-dag-path data/spec_datasets/nist23/subformulae/no_subform.hdf5 \
    --out-dag-path results/marason_nist23/${folder}/preds_train_100_inten.hdf5 \
    --num-workers 32
done
