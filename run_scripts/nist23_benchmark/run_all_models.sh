#!/usr/bin/env bash
# NIST'23 benchmark — train + predict + evaluate every model, then aggregate.
#
# Prerequisite: run_scripts/nist23_benchmark/00_preprocess.sh has completed.
#
# Usage:
#   bash run_scripts/nist23_benchmark/run_all_models.sh              # all 6 models
#   bash run_scripts/nist23_benchmark/run_all_models.sh iceberg glacier   # subset
#
# Each model runs its full nist23 pipeline for the two benchmark splits
# (random split_1 + scaffold_1, seed 1 each). Models are independent — run them
# on separate GPUs / sessions if you want to parallelise. This driver runs them
# sequentially and is safe to re-run (training resumes / predict overwrites).
set -euo pipefail

MODELS=("$@")
if [[ ${#MODELS[@]} -eq 0 ]]; then
  MODELS=(massformer molnetms glacier iceberg scarf marason)
fi

run_massformer() {
  echo "### MassFormer ###"
  bash run_scripts/massformer_model/nist23/01_train.sh
  python run_scripts/massformer_model/nist23/02_predict.py
}

run_molnetms() {
  echo "### 3DMolMS ###"
  bash run_scripts/molnetms/nist23/01_train.sh
  python run_scripts/molnetms/nist23/02_predict.py
}

run_glacier() {
  echo "### GLACIER ###"
  bash run_scripts/glacier/nist23/01_train.sh
  python run_scripts/glacier/nist23/02_predict.py
}

run_iceberg() {
  echo "### ICEBERG 2.1 ###"
  bash run_scripts/iceberg/nist23/01_run_dag_gen_train.sh
  bash run_scripts/iceberg/nist23/03_run_dag_gen_predict.sh
  bash run_scripts/iceberg/nist23/04_train_dag_inten.sh
  python run_scripts/iceberg/nist23/05_predict_dag_inten.py
}

run_scarf() {
  echo "### SCARF ###"
  bash run_scripts/scarf_model/nist23/01_gen_train.sh
  bash run_scripts/scarf_model/nist23/02_gen_predict.sh
  bash run_scripts/scarf_model/nist23/03_inten_train.sh
  python run_scripts/scarf_model/nist23/04_predict.py
}

run_marason() {
  echo "### MARASON ###"
  bash run_scripts/marason/nist23/01_gen_train.sh
  bash run_scripts/marason/nist23/02_gen_predict.sh
  bash run_scripts/marason/nist23/03_inten_train.sh
  python run_scripts/marason/nist23/04_predict.py
}

for m in "${MODELS[@]}"; do
  case "$m" in
    massformer) run_massformer ;;
    molnetms|3dmolms) run_molnetms ;;
    glacier) run_glacier ;;
    iceberg) run_iceberg ;;
    scarf) run_scarf ;;
    marason) run_marason ;;
    *) echo "unknown model: $m (expected one of: massformer molnetms glacier iceberg scarf marason)"; exit 1 ;;
  esac
done

echo "### Aggregating results ###"
python analysis/nist23_benchmark_aggregate.py \
  --results-dir results \
  --out-tsv results/nist23_benchmark.tsv \
  --out-md results/nist23_benchmark.md
