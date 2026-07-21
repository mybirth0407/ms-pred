#!/usr/bin/env bash
# SCARF NIST'23 — stage 2: predict formula subsets for the training molecules and
# attach intensities so stage 3 (intensity model) can train.
#
# !!! KNOWN UPSTREAM ISSUE (affects nist20 identically, not specific to nist23) !!!
# src/ms_pred/scarf_pred/predict_gen.py writes per-spectrum JSON files into a
# `.../form_preds/` DIRECTORY, but data_scripts/forms/03_add_form_intens.py opens
# its --pred-form-folder with common.HDF5Dataset (i.e. expects a single .hdf5 file).
# The ICEBERG accuracy PR (#19) migrated add_form_intens.py to HDF5 but predict_gen.py
# was not migrated. If the add_form_intens step below fails opening `form_preds`,
# the SCARF gen->inten plumbing needs an upstream fix (make predict_gen.py emit an
# HDF5 store, or teach add_form_intens.py to read a JSON directory). Verify against
# your data before trusting SCARF numbers.
set -euo pipefail

python launcher_scripts/run_from_config.py configs/scarf/nist23/scarf_gen_predict_train_nist23.yaml

for folder in split_1_rnd1 scaffold_1_rnd1; do
  python data_scripts/forms/03_add_form_intens.py \
    --pred-form-folder results/scarf_nist23/${folder}/preds_train_300/form_preds \
    --true-form-folder data/spec_datasets/nist23/subformulae/no_subform.hdf5 \
    --out-form-folder results/scarf_nist23/${folder}/preds_train_300_inten \
    --num-workers 16 \
    --add-raw \
    --binned-add
done
