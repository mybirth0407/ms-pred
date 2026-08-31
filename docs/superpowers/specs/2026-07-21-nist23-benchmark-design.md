# NIST23 Spectrum-Prediction Benchmark — Design

**Date:** 2026-07-21
**Goal:** Benchmark every trainable model in this repo on the NIST'23 dataset and
report spectrum-prediction accuracy in one comparable table.

## Scope (decided with user)

- **Models (6):** ICEBERG 2.1, SCARF, MassFormer, 3DMolMS (MolNetMS), GLACIER, MARASON.
  These are all the neural models in the repo that train + predict on an NIST-style
  dataset. Excluded by the user: retrieval baselines (CFM-ID, MetFrag, freq/random),
  NEIMS-FFN/GNN and autoregr baseline (not requested — spectrum-model comparison only).
- **Data source:** raw NIST'23 `.SDF` (path provided by user at run time).
- **Splits:** `split_1` (random) and `scaffold_1`, **seed 1 each** → two runs per model.
  Split TSVs already exist in `data/spec_datasets/nist23/splits/` (176,851 spectra:
  train 142k / val 16k / test 18k).
- **Metric:** spectrum-prediction accuracy only (cosine similarity, entropy similarity,
  coverage) via `analysis/spec_pred_eval.py`. **No retrieval.**
- **Intermediate eval:** skipped (no gen-threshold sweeps / fragment-count trade-offs).
- **Contrastive finetuning:** OFF by default. It exists to improve *retrieval* ranking
  (trains against PubChem decoys) and needs `pubchem_formulae_inchikey.hdf5` + retrieval
  candidate tables — both out of scope. Turning it off keeps all 6 models on equal
  footing (pure forward spectrum prediction) and avoids extra downloads. Kept available
  and documented as opt-in.

## Environment

Isolated `.venv` built with `uv sync --extra cu124 --extra test --python 3.12`
(Python **must** be 3.12 — `torchdata<0.10` has no cp313 wheel; base conda is 3.13 and
also shadows `ms_pred` with a different workspace, so base must not be used).
Verified: torch 2.6.0+cu124, dgl 2.5.0+cu124, 8× RTX 6000 Ada, `ms_pred` → this repo.

## Data pipeline (shared, run once)

1. Convert NIST'23 `.SDF` → `labels.tsv` + `spec_files.hdf5` + `mgf_files/` using the
   external `ms-data-parser` (`reformat_nist_lcmsms_sdf.py`), output into
   `data/spec_datasets/nist23/`.
2. **Critical validation:** spectrum IDs from the parser must match the existing split
   TSVs (`nist_1035166` form). If they don't, the pre-shipped splits are unusable.
   `00_preprocess.sh` asserts overlap before continuing.
3. Subformulae: `no_subform.hdf5` (permissive) + `magma_subform_50.hdf5` (SCARF).
4. MAGMa fragment labels via `data_scripts/dag/run_magma.sh nist23` (needed by
   ICEBERG / GLACIER / MARASON). GLACIER additionally needs `add_dag_intens` on the
   MAGMa tree.

## Per-model pipeline

| Model | Stages |
|-------|--------|
| MassFormer | train → predict+eval (binned, single stage) |
| 3DMolMS | train → predict+eval (binned, single stage) |
| GLACIER | joint train → predict+eval |
| ICEBERG | gen train → gen predict (+add intens) → inten train → predict+eval |
| SCARF | gen train → gen predict (+add form intens) → inten train → predict+eval |
| MARASON | gen train → gen predict (+add intens) → inten train (RAG) → predict+eval |

All predict steps route through `analysis/spec_pred_eval.py`, which writes
`preds/pred_eval.yaml` with `avg_cos_sim`, `avg_entropy_sim`, `avg_coverage`.

## Deliverables

Mirror the existing `run_scripts/iceberg/nist23/` + `configs/iceberg/nist23/` convention
for the 5 models that lack it, transformed to the scope above:

- `configs/<model>/nist23/*.yaml` for scarf, massformer, 3dmolms, glacier, marason.
- `run_scripts/<model>/nist23/*` for the same, with predict drivers trimmed to the two
  splits, checkpoint versions corrected (ICEBERG → `ckpt/inten`, GLACIER → `version_0`),
  and the MassFormer `test_dataset` KeyError bug fixed.
- ICEBERG: activate `scaffold_1_rnd1`, set `num-decoys: [0]`, point predict to `ckpt/inten`.
- Top level `run_scripts/nist23_benchmark/`:
  - `00_preprocess.sh` — SDF→hdf5, split-ID assertion, subformulae, MAGMa.
  - `run_all_models.sh` — runs all 6 model pipelines.
  - `aggregate_results.py` — collects every `pred_eval.yaml` into one TSV/markdown table.
  - `README.md` — how to point at the NIST23 SDF and run.
- Bug fixes in existing scripts: `run_scripts/glacier/run_all.sh` (`GLACIER`→`glacier`
  casing) and `glacier/add_inten.sh` (add nist23).

## Validation before handing off

Every generated config is checked with `python launcher_scripts/run_from_config.py <cfg> --dry`
(no data needed) to confirm it parses and expands to the expected command list.

## Compute note

142k training spectra × 6 models × 2 splits on 8× RTX 6000 Ada is multi-day wall-clock.
`run_all_models.sh` is staged so models can be launched independently / incrementally.
