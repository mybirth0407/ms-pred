# NIST'23 spectrum-prediction benchmark

Trains and evaluates every trainable spectrum model in this repo on **NIST'23** and
reports spectrum-prediction accuracy (cosine similarity, entropy similarity, coverage)
in one comparable table.

**Models (6):** ICEBERG 2.1, SCARF, MassFormer, 3DMolMS, GLACIER, MARASON.
**Splits:** `split_1` (random) and `scaffold_1`, seed 1 each — 2 runs per model.
**Scope:** spectrum prediction only. No retrieval, no contrastive finetuning
(see *Scope decisions* below). Design doc:
[`docs/superpowers/specs/2026-07-21-nist23-benchmark-design.md`](../../docs/superpowers/specs/2026-07-21-nist23-benchmark-design.md).

## 0. Environment

The repo pins `torch==2.6.0+cu124` and `torchdata<0.10` (no cp313 wheel), so the env
**must** be Python 3.12 — do not use the base conda (3.13, and it shadows `ms_pred`
with another workspace):

```bash
uv sync --extra cu124 --extra test --python 3.12
source .venv/bin/activate
python -c "import torch, dgl, ms_pred; print(torch.__version__, dgl.__version__, ms_pred.__file__)"
```

## 1. Point at the NIST'23 data

`data/spec_datasets/nist23/splits/{split_1,scaffold_1}.tsv` already ship with the repo.
You provide the spectra. Build `labels.tsv` + `spec_files.hdf5` with the external
[ms-data-parser](https://github.com/rogerwwww/ms-data-parser)'s
`reformat_nist_lcmsms_sdf.py`, which needs a **combined SDF** (molblocks carrying the
spectral `> <FIELD>` blocks).

Two export shapes:

**(a) Combined SDF** (already has `<MASS SPECTRAL PEAKS>`, `<NISTNO>`, `<PRECURSOR TYPE>`…):

```bash
git clone https://github.com/rogerwwww/ms-data-parser
python ms-data-parser/reformat_nist_lcmsms_sdf.py \
    --input-file /path/to/combined_nist23.sdf \
    --targ-dir data/spec_datasets/nist23 --dataset nist2023 --workers 32
```

**(b) Structure-only `.sdf` + spectra `.msp`** (some NIST'23 exports split these — the
structure SDF has molblocks but no peaks; the MSP has peaks/adduct/collision-energy/
InChIKey but no structure). Merge them first, then reformat:

```bash
python data_scripts/nist23/build_combined_sdf.py \
    --raw-dir /path/to/nist23_export --out /tmp/combined_nist23.sdf
python ms-data-parser/reformat_nist_lcmsms_sdf.py \
    --input-file /tmp/combined_nist23.sdf \
    --targ-dir data/spec_datasets/nist23 --dataset nist2023 --workers 32
```

`build_combined_sdf.py` joins structures to spectra by RDKit InChIKey (exact stereo,
connectivity fallback) — robust to the SDF/MSP ordering indels and name-encoding
mismatches. `00_preprocess.sh` runs this path automatically when `NIST23_RAW` and
`MS_DATA_PARSER` are set. Result must look like:

```
data/spec_datasets/nist23/
├── labels.tsv
├── mgf_files/
├── spec_files.hdf5
└── splits/{split_1.tsv, scaffold_1.tsv}
```

> **Critical:** the parser's spectrum IDs must match the split files (`nist_<n>` form).
> `00_preprocess.sh` asserts this and aborts if they diverge. Our NIST'23 build
> reproduced 99.4% of the shipped split IDs.

## 2. Preprocess (once)

```bash
bash run_scripts/nist23_benchmark/00_preprocess.sh
```

Builds subformulae (`no_subform`, `magma_subform_50`), MAGMa fragment DAGs, and the
GLACIER intensity-augmented tree. Idempotent — re-running skips finished stages.

## 3. Run the benchmark

```bash
# all six models, sequentially, then aggregate
bash run_scripts/nist23_benchmark/run_all_models.sh

# or a subset (e.g. run the two cheapest single-stage models first)
bash run_scripts/nist23_benchmark/run_all_models.sh massformer molnetms
```

Each model trains → predicts on the test set → evaluates. Models are independent; put
them on different GPUs / sessions to parallelise. Training 142k spectra × 6 models ×
2 splits is multi-day on 8× RTX 6000 Ada, so staged/incremental runs are recommended.

## 4. Results

Every predict step writes `results/<model>_nist23/<split>/preds/pred_eval.yaml`.
Aggregate them any time (also run automatically at the end of `run_all_models.sh`):

```bash
python analysis/nist23_benchmark_aggregate.py
# -> results/nist23_benchmark.tsv and results/nist23_benchmark.md
```

## Scope decisions

- **No contrastive finetuning.** It trains against PubChem decoys to improve *retrieval*
  ranking; the user scoped this benchmark to spectrum accuracy without retrieval. Turning
  it off keeps all models on equal footing and avoids the PubChem-map download. To enable
  it for ICEBERG/GLACIER, see the comments in their `nist23` train scripts.
- **Two splits, one seed.** `split_1_rnd1` + `scaffold_1_rnd1`. To reproduce the papers'
  full grid (3 seeds × 2 splits), add the extra `iterative_args` entries in each
  `configs/<model>/nist23/*.yaml` and the matching `test_entries` in the predict drivers.

## Known upstream issue (SCARF)

`src/ms_pred/scarf_pred/predict_gen.py` writes per-spectrum JSON into a `form_preds/`
**directory**, but `data_scripts/forms/03_add_form_intens.py` opens `--pred-form-folder`
as an **HDF5 file** (the ICEBERG accuracy PR migrated add_form_intens to HDF5 but not
predict_gen). This affects the `nist20` scripts identically. If SCARF stage 2
(`02_gen_predict.sh`) fails at the `add_form_intens` step, predict_gen needs an upstream
fix to emit an HDF5 store (or add_form_intens needs to read a JSON directory). Verify
against your data before trusting SCARF numbers.
