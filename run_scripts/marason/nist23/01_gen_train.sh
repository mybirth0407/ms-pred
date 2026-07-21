#!/usr/bin/env bash
# MARASON NIST'23 — stage 1: train the fragment generator (same DAG generator as
# ICEBERG). Random split_1 + scaffold_1, seed 1 each.
python launcher_scripts/run_from_config.py configs/marason/nist23/marason_train_nist23.yaml
