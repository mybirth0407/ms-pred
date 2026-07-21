#!/usr/bin/env bash
# GLACIER NIST'23 — joint train (random split_1 + scaffold_1, seed 1 each).
# Contrastive finetuning is intentionally omitted for the spectrum-only benchmark
# (it needs PubChem decoys + retrieval tables). To enable it, create a nist23
# variant of configs/glacier/joint_contr_finetune_nist20.yaml and run it here.
python launcher_scripts/run_from_config.py configs/glacier/nist23/joint_train_nist23.yaml
