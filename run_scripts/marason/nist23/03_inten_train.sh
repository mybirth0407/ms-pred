#!/usr/bin/env bash
# MARASON NIST'23 — stage 3: train the RAG-based intensity model.
# The config uses add-reference/load-reference/save-reference = false, so
# reference neighbours are computed on the fly from the training set (no
# precomputed data/closest_neighbors store required for training).
python launcher_scripts/run_from_config.py configs/marason/nist23/marason_inten_train_nist23.yaml
