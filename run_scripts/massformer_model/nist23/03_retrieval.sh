#!/usr/bin/env bash
# MassFormer NIST'23 retrieval (candidate ranking) — full reproduction.
#
# WHY THIS IS NON-TRIVIAL:
#   * predict.py must write PredSpecDB-format binned spectra (not the legacy raw
#     HDF5 layout) so retrieval_benchmark.py's PredSpecDB reader can consume them.
#     (Fixed in src/ms_pred/massformer_pred/predict.py.)
#   * The candidate set is huge: ~820k (spec, candidate) rows x ~6.6 collision
#     energies ~= 5.4M binned spectra. Writing >~200k entries into a single HDF5
#     from one process blows up RSS (HDF5 object growth) -> OOM. So we split the
#     candidates into 32 spec-balanced chunks (~170k entries each) and predict
#     them 8-GPU-parallel into binned_preds_shard{0..31}.hdf5. PredSpecDB
#     auto-merges sibling *_shard*.hdf5 files on read (scan-fallback locates each
#     spec regardless of chunk), so no explicit merge step is needed.
#
# Runtime: ~50 min predict (8x RTX 6000 Ada) + ~45 min rerank. Peak RAM ~160 GB.
set -euo pipefail

REPO=/home/mybirth0407/workspaces/ms-pred
PY="$REPO/.venv/bin/python"
DATASET=nist23
SPLIT=scaffold_1
MAXK=50
NCHUNK=32
CKPT=results/massformer_baseline_${DATASET}/scaffold_1_rnd1/version_0/best.ckpt
SAVE_ROOT=results/massformer_baseline_${DATASET}/scaffold_1_rnd1/retrieval_${DATASET}_${SPLIT}_${MAXK}
CANDS=data/spec_datasets/${DATASET}/retrieval/cands_df_${SPLIT}_${MAXK}.tsv
WORK="$SAVE_ROOT/_chunks"

cd "$REPO"
mkdir -p "$SAVE_ROOT" "$WORK"

# --- STEP 0: split candidates into NCHUNK spec-balanced chunks -----------------
"$PY" - "$CANDS" "$WORK" "$NCHUNK" <<'PY'
import ast, sys
from pathlib import Path
import pandas as pd
cands, work, n = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3])
df = pd.read_csv(cands, sep="\t")
def nce(x):
    try: return max(1, len([c for c in ast.literal_eval(x) if "nan" not in str(c)]))
    except Exception: return 1
w = (df.groupby("spec").size() *
     df.groupby("spec")["collision_energies"].first().map(nce)).sort_values(ascending=False)
loads=[0]*n; assign={}
for spec,ww in w.items():
    i=min(range(n), key=lambda k: loads[k]); assign[spec]=i; loads[i]+=int(ww)
df["_c"]=df["spec"].map(assign)
for i in range(n):
    df[df["_c"]==i].drop(columns="_c").to_csv(work/f"cands_chunk{i}.tsv", sep="\t", index=False)
print(f"split into {n} chunks; item-weight spread={max(loads)-min(loads)}")
PY

# --- STEP 1: predict all candidates, 8-GPU-parallel, one shard file per chunk --
rm -f "$SAVE_ROOT"/binned_preds_shard*.hdf5
i=0
while [ $i -lt $NCHUNK ]; do
  pids=(); idxs=()
  for gpu in 0 1 2 3 4 5 6 7; do
    [ $i -lt $NCHUNK ] || break
    CUDA_VISIBLE_DEVICES=$gpu "$PY" src/ms_pred/massformer_pred/predict.py \
      --batch-size 32 --num-workers 4 --dataset-name $DATASET \
      --sparse-out --sparse-k 100 --split-name ${SPLIT}.tsv \
      --num-bins 15000 --upper-limit 1500 \
      --checkpoint-pth "$CKPT" \
      --save-dir "$SAVE_ROOT" --out-name "binned_preds_shard${i}.hdf5" \
      --dataset-labels "$WORK/cands_chunk${i}.tsv" \
      --gpu > "$WORK/predict_chunk${i}.log" 2>&1 &
    pids+=($!); idxs+=($i); i=$((i+1))
  done
  echo "predict wave: chunks ${idxs[*]}"
  for j in "${!pids[@]}"; do wait "${pids[$j]}" || { echo "chunk ${idxs[$j]} FAILED"; exit 1; }; done
done

# --- STEP 2: rank candidates -> top-k (entropy, matching the fragment models) --
"$PY" src/ms_pred/retrieval/retrieval_benchmark.py \
  --dataset $DATASET \
  --formula-dir-name no_subform.hdf5 \
  --pred-file "$SAVE_ROOT/binned_preds.hdf5" \
  --full-labels "$CANDS" \
  --dist-fn entropy --num-bins 15000 --upper-limit 1500 \
  --num-cpu-workers 32

echo "=== top-k ==="
grep -E "^avg_top_[0-9]+:" "$SAVE_ROOT/rerank_eval_entropy.yaml"
