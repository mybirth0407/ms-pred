# NIST'23 Spectrum-Prediction Benchmark — Results

**Dataset:** NIST'23 (176,697 spectra / 47,203 molecules)
**Split:** Bemis–Murcko **scaffold** split (`scaffold_1.tsv`), seed 1 — no molecular leakage across folds (verified: 0 inchikey/connectivity spanning >1 fold)
**Test set:** 17,567 spectra / 9,291 molecules
**Eval:** `analysis/spec_pred_eval.py`, `--max-peaks 100 --min-inten 0`, gold = `no_subform.hdf5`
**Scope:** spectrum-prediction accuracy only; no test-time retrieval augmentation; contrastive finetuning off.

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

## 3. Reproduce

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

## 4. Notes / caveats
- All four models train single-GPU except GLACIER (multi-GPU DDP); GLACIER was run on 4 GPUs.
- Fragment-model evaluation (ICEBERG / MARASON / GLACIER) requires the numpy≥2 `safe_assign` fix and, for GLACIER, the `lr_scheduler_step(metric=None)` + `ddp_find_unused_parameters_true` fixes.
- Not in this batch: **SCARF** (needs `magma_subform_50`), **3DMolMS** (separate run). The aggregator/table slots exist for both.
