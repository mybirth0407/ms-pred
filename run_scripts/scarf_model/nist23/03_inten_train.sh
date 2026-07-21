#!/usr/bin/env bash
# SCARF NIST'23 — stage 3: train the intensity model on the generated formula subsets.
python launcher_scripts/run_from_config.py configs/scarf/nist23/scarf_inten_train_nist23.yaml
