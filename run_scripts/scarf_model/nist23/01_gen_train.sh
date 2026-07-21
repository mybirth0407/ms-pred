#!/usr/bin/env bash
# SCARF NIST'23 — stage 1: train the prefix-tree formula generator
# (random split_1 + scaffold_1, seed 1 each).
python launcher_scripts/run_from_config.py configs/scarf/nist23/scarf_train_nist23.yaml
