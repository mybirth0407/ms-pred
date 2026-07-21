#!/usr/bin/env bash
# NIST'23 benchmark — shared data preprocessing (run once, before any model).
#
# Produces everything the 6 models need under data/spec_datasets/nist23/:
#   labels.tsv, spec_files.hdf5            (from the raw NIST'23 SDF; see stage 0)
#   subformulae/no_subform.hdf5            (permissive subformulae — all models)
#   subformulae/magma_subform_50.hdf5      (MAGMa-labelled subformulae — SCARF)
#   subformulae/magma_subform_50_with_raw.hdf5
#   magma_outputs/magma_tree.hdf5          (MAGMa fragment DAGs — ICEBERG/GLACIER/MARASON)
#   magma_outputs/magma_tree_with_inten.hdf5 (GLACIER only)
#
# Usage:
#   NIST23_SDF=/path/to/nist23.sdf bash run_scripts/nist23_benchmark/00_preprocess.sh
# If labels.tsv + spec_files.hdf5 already exist, stage 0 is skipped and NIST23_SDF
# is not required.
set -euo pipefail

DATASET=nist23
DATA_DIR="data/spec_datasets/${DATASET}"
LABELS="${DATA_DIR}/labels.tsv"
SPECS="${DATA_DIR}/spec_files.hdf5"
WORKERS="${WORKERS:-32}"
PPM_DIFF="${PPM_DIFF:-20}"

mkdir -p "${DATA_DIR}"

# ---------------------------------------------------------------------------
# Stage 0: raw SDF -> labels.tsv + spec_files.hdf5 (+ mgf_files/)
# This uses the external NIST parser (reformat_nist_lcmsms_sdf.py) from
# https://github.com/rogerwwww/ms-data-parser — it is NOT vendored in this repo.
# ---------------------------------------------------------------------------
if [[ -f "${LABELS}" && -f "${SPECS}" ]]; then
  echo "[stage 0] ${LABELS} and ${SPECS} already exist — skipping SDF conversion."
else
  echo "[stage 0] ERROR: ${LABELS} / ${SPECS} not found."
  echo "  Convert the raw NIST'23 .SDF first with the external ms-data-parser, e.g.:"
  echo "    git clone https://github.com/rogerwwww/ms-data-parser"
  echo "    python ms-data-parser/reformat_nist_lcmsms_sdf.py \\"
  echo "        --input \"\${NIST23_SDF:?set NIST23_SDF to the exported .SDF}\" \\"
  echo "        --output-dir ${DATA_DIR}"
  echo "  The output must land as ${DATA_DIR}/{labels.tsv,spec_files.hdf5,mgf_files/}."
  exit 1
fi

# ---------------------------------------------------------------------------
# Stage 1: CRITICAL — verify parsed spectrum IDs match the shipped split TSVs.
# The split files (data/spec_datasets/nist23/splits/*.tsv) are pre-generated with
# IDs like `nist_1035166`. If the parser emits different IDs the splits are unusable
# and every model would train/test on the wrong (or empty) data.
# ---------------------------------------------------------------------------
echo "[stage 1] validating spectrum IDs against split files..."
python - "$LABELS" "${DATA_DIR}/splits/split_1.tsv" "${DATA_DIR}/splits/scaffold_1.tsv" <<'PY'
import sys, csv
from pathlib import Path

labels_path, *split_paths = sys.argv[1:]

def first_col_ids(path, has_header=True):
    ids = set()
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        if has_header:
            next(reader, None)
        for row in reader:
            if row:
                ids.add(row[0])
    return ids

# labels.tsv: find the spec-id column ("spec" is the split key; labels use "spec"/"spec_id")
with open(labels_path, newline="") as fh:
    header = next(csv.reader(fh, delimiter="\t"))
id_col = None
for cand in ("spec", "spec_id", "spectrum_id"):
    if cand in header:
        id_col = header.index(cand)
        break
if id_col is None:
    id_col = 0
label_ids = set()
with open(labels_path, newline="") as fh:
    reader = csv.reader(fh, delimiter="\t")
    next(reader, None)
    for row in reader:
        if row:
            label_ids.add(row[id_col])

print(f"  labels.tsv: {len(label_ids)} spectra (id column '{header[id_col]}')")
ok = True
for sp in split_paths:
    split_ids = first_col_ids(sp)
    overlap = split_ids & label_ids
    frac = len(overlap) / max(len(split_ids), 1)
    status = "OK" if frac > 0.99 else "MISMATCH"
    if frac <= 0.99:
        ok = False
    print(f"  {Path(sp).name}: {len(split_ids)} ids, {len(overlap)} present in labels "
          f"({frac:.1%}) -> {status}")

if not ok:
    print("\n  ABORT: split IDs do not match parsed labels. The shipped splits assume "
          "IDs of the form 'nist_<n>'. Re-run the parser so labels.tsv uses the same "
          "IDs, or regenerate splits for your IDs before continuing.")
    sys.exit(2)
print("  split validation passed.")
PY

# ---------------------------------------------------------------------------
# Stage 2: subformulae (no_subform + magma_subform_50 + with_raw)
# ---------------------------------------------------------------------------
echo "[stage 2] assigning subformulae..."
if [[ -f "${DATA_DIR}/subformulae/no_subform.hdf5" ]]; then
  echo "  no_subform.hdf5 exists — skipping."
else
  python data_scripts/forms/01_assign_subformulae.py \
    --data-dir "${DATA_DIR}/" \
    --labels-file "${LABELS}" \
    --use-all \
    --output-dir-name no_subform.hdf5 \
    --num-workers "${WORKERS}"
fi

if [[ -f "${DATA_DIR}/subformulae/magma_subform_50.hdf5" ]]; then
  echo "  magma_subform_50.hdf5 exists — skipping."
else
  python data_scripts/forms/01_assign_subformulae.py \
    --data-dir "${DATA_DIR}/" \
    --labels-file "${LABELS}" \
    --use-magma \
    --mass-diff-thresh "${PPM_DIFF}" \
    --output-dir-name magma_subform_50.hdf5 \
    --num-workers "${WORKERS}"
fi

if [[ -f "${DATA_DIR}/subformulae/magma_subform_50_with_raw.hdf5" ]]; then
  echo "  magma_subform_50_with_raw.hdf5 exists — skipping."
else
  python data_scripts/forms/03_add_form_intens.py \
    --num-workers "${WORKERS}" \
    --pred-form-folder "${DATA_DIR}/subformulae/magma_subform_50.hdf5" \
    --true-form-folder "${DATA_DIR}/subformulae/no_subform.hdf5" \
    --add-raw \
    --binned-add \
    --out-form-folder "${DATA_DIR}/subformulae/magma_subform_50_with_raw.hdf5"
fi

# ---------------------------------------------------------------------------
# Stage 3: MAGMa fragment DAG labels (ICEBERG / GLACIER / MARASON)
# run_magma.sh defaults to nist23 and also (re)creates no_subform if missing.
# ---------------------------------------------------------------------------
echo "[stage 3] MAGMa fragment labelling..."
if [[ -f "${DATA_DIR}/magma_outputs/magma_tree.hdf5" ]]; then
  echo "  magma_tree.hdf5 exists — skipping."
else
  WORKERS="${WORKERS}" ppm_diff="${PPM_DIFF}" bash data_scripts/dag/run_magma.sh "${DATASET}"
fi

# ---------------------------------------------------------------------------
# Stage 4: GLACIER intensity-augmented MAGMa tree
# ---------------------------------------------------------------------------
echo "[stage 4] GLACIER magma_tree_with_inten..."
if [[ -f "${DATA_DIR}/magma_outputs/magma_tree_with_inten.hdf5" ]]; then
  echo "  magma_tree_with_inten.hdf5 exists — skipping."
else
  bash run_scripts/glacier/nist23/00_add_inten.sh
fi

echo "[done] NIST'23 preprocessing complete."
