# NIST'23 Spectrum-Prediction Benchmark — Results

**Dataset:** NIST'23 (176,697 spectra / 47,203 molecules)
**Split:** Bemis–Murcko **scaffold** split (`scaffold_1.tsv`), seed 1 — no molecular leakage across folds (verified: 0 inchikey/connectivity spanning >1 fold)
**Test set:** 17,567 spectra / 9,291 molecules
**Eval:** `analysis/spec_pred_eval.py`, `--max-peaks 100 --min-inten 0`, gold = `no_subform.hdf5`
**Scope:** spectrum-prediction accuracy only; no test-time retrieval augmentation; contrastive finetuning off.
**Benchmark run:** 2026-07-22 → 2026-07-23 (all training + evaluation; dates below are result-file timestamps).

## Run dates

| Model | Trained | Spectrum eval | Stage-1 eval | Retrieval |
|------|:---:|:---:|:---:|:---:|
| GLACIER | 2026-07-22 13:51 | 2026-07-22 18:19 | — | 2026-07-23 15:48 |
| ICEBERG 2.1 | 2026-07-22 20:32 | 2026-07-23 03:04 | 2026-07-23 06:57 | 2026-07-23 11:21 |
| MARASON | 2026-07-22 23:18 | 2026-07-23 06:01 | 2026-07-23 07:00 | 2026-07-23 22:18 |
| MassFormer | 2026-07-22 15:00 | 2026-07-22 17:19 | — | 2026-07-29 05:00 |

- **Trained** = `best.ckpt` save time. For the 2-stage models this is the **stage-2 (inten)** checkpoint; stage-1 (gen) finished earlier — ICEBERG 07-22 05:46, MARASON 07-22 05:47.
- **Eval** = result-yaml write time. `best.ckpt` marks the best-epoch save, so it can precede the actual end of training.
- GLACIER / MassFormer have no Stage-1 run (`—`). MassFormer retrieval was added later (2026-07-29) — it needed a `predict.py` fix (see §Retrieval).

## 1. Spectrum accuracy (test)

Ranked by cosine similarity. SEM ≈ 0.0005–0.0009 for all (large test set).

| Rank | Model | Type | Cosine | Entropy | Coverage |
|:---:|------|------|:---:|:---:|:---:|
| 1 | **GLACIER** | joint (multi-GPU DDP) | **0.800** | 0.736 | 0.868 |
| 2 | **ICEBERG 2.1** | 2-stage fragment | **0.722** | 0.647 | 0.818 |
| 3 | **MARASON** | 2-stage fragment + RAG¹ | **0.720** | 0.646 | 0.805 |
| 4 | **MassFormer** | binned graph transformer | **0.512** | 0.486 | 0.710 |

- **Cosine / Entropy** = full-spectrum similarity (peak position **and** intensity).
- **Coverage** = fraction of true peaks whose m/z the prediction hit (position-only recall).
- Fragment-based models (GLACIER / ICEBERG / MARASON) clearly outperform the binned model (MassFormer).

¹ **MARASON is evaluated in base mode (no retrieval augmentation)** — it was trained with on-the-fly references (`add-reference=false`) and the `--add-ref` eval path needs a precomputed nearest-neighbour store that is not yet built. Base MARASON ≈ ICEBERG (same generator family). The RAG-augmented number requires building the reference store (part of the retrieval task).

## 2. Stage-1 fragment-DAG quality (2-stage models, test)

How well the stage-1 **generator** recovers the true MAGMa fragment set, before intensity prediction.
Fragment identity = Weyl–Leman graph hash (`--identity wl`, matches the original MAGMa `frag_hash`). Computed by `analysis/dag_frag_eval.py` on the test split.

| Model | Precision | Recall | F1 | Jaccard | pred frags | true frags |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| ICEBERG 2.1 | 0.265 | **0.774** | 0.357 | 0.228 | 95.4 | 37.1 |
| MARASON | 0.265 | **0.775** | 0.357 | 0.229 | 95.4 | 37.1 |

- **Recall ≈ 0.77** — the generator recovers ~77% of true fragments (good completeness on unseen molecules).
- **Precision ≈ 0.27** — of ~95 generated fragments only ~27% are real (37 true on average); expected at threshold 0.0 / max-nodes 100. Lower `max-nodes` / higher `threshold` trades recall for precision (precision–recall curve).
- ICEBERG ≈ MARASON — they share the same DAG generator architecture.

## 3. Retrieval top-k (candidate ranking, test)

Rank each spectrum's candidate set (`cands_df_scaffold_1_50.tsv`, ≤50 true+PubChem-decoy per spec) by distance between predicted and observed spectra. All four models: same 17,565 test spectra, same candidate set, **entropy** distance, `num-bins 15000`.

| Model | top-1 | top-2 | top-3 | top-5 | top-10 | top-20 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **GLACIER** | **0.396** | 0.578 | 0.683 | 0.796 | **0.905** | 0.968 |
| ICEBERG 2.1 | 0.345 | 0.526 | 0.638 | 0.768 | 0.901 | 0.967 |
| MARASON | 0.339 | 0.519 | 0.634 | 0.761 | 0.891 | 0.964 |
| MassFormer | 0.214 | 0.359 | 0.461 | 0.600 | 0.780 | 0.916 |

- Same accuracy ranking as §1 — better spectrum prediction → better candidate ranking. MassFormer is weakest across all k.
- MassFormer: all 17,565 specs scored, 0 missing candidates; `avg_total_decoys=46.7` (= the other models' — not the per-CE-inflated GLACIER count), so top-k is directly comparable.
- **MassFormer required a `predict.py` fix.** It previously wrote a legacy raw-HDF5 layout that `retrieval_benchmark.py` (PredSpecDB reader) cannot parse; `predict.py` now writes PredSpecDB-format binned spectra (like ICEBERG). The ~5.4M candidate spectra are predicted in 32 spec-balanced shards to bound HDF5 write-side RAM. Repro: `run_scripts/massformer_model/nist23/03_retrieval.sh`.

## 4. Reproduce

Per-model eval outputs live at `results/<model>_nist23/scaffold_1_rnd1/preds/pred_eval.yaml`
(+ grouped TSVs by collision energy / ion type / mass bin).
Stage-1 fragment metrics: `results/<model>_nist23/scaffold_1_rnd1/pred_eval_frag_test.yaml`.

```bash
# Spectrum table (all models with a pred_eval.yaml):
python analysis/nist23_benchmark_aggregate.py

# Stage-1 fragment P/R/F1 (ICEBERG + MARASON, test-only, CPU):
python analysis/dag_frag_eval.py \
  --pred-tree-h5 results/<model>_nist23/scaffold_1_rnd1/preds_train_100/tree_preds.hdf5 \
  --gold-tree-h5 data/spec_datasets/nist23/magma_outputs/magma_tree.hdf5 \
  --dataset nist23 --identity wl --split-name scaffold_1.tsv --subset test --max-cpu 16
```

## 5. Notes / caveats
- All four models train single-GPU except GLACIER (multi-GPU DDP); GLACIER was run on 4 GPUs.
- Fragment-model evaluation (ICEBERG / MARASON / GLACIER) requires the numpy≥2 `safe_assign` fix and, for GLACIER, the `lr_scheduler_step(metric=None)` + `ddp_find_unused_parameters_true` fixes.
- Not in this batch: **SCARF** (needs `magma_subform_50`), **3DMolMS** (separate run). The aggregator/table slots exist for both.
