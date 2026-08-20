"""MAGMa oracle upper bound for fragment/subformula spectrum models.

WHAT IT MEASURES
----------------
The theoretical ceiling that MAGMa's fragment enumeration imposes: if a model
assigned the *true* intensity to every MAGMa-enumerated fragment (and could place
intensity nowhere else), how well would that reconstruct the FULL true spectrum?
Peaks/intensity that no MAGMa fragment can reach are the hard ceiling.

  oracle_spectrum(spec) = gold spectrum  masked to MAGMa-reachable m/z bins
  cos@k / coverage      = analysis/spec_pred_eval.py logic (top-k pred peaks, min_inten 0)

IMPORTANT SCOPE
---------------
* This ceiling is a property of the MAGMa fragment set + the true spectra, NOT of a
  trained model's weights. On a FIXED (spec, CE) set with a FIXED reachable-set
  convention, every fragment model shares the same oracle number.
* It is the ceiling ONLY for fragment/subformula models (ICEBERG, GLACIER, MARASON,
  SCARF). Binned models (MassFormer, 3DMolMS) predict arbitrary bins, so their ceiling
  is 1.0 — this oracle does NOT bound them.
* The only model dependence is the fragment->m/z CONVENTION:
    - "both" = adduct-shift group + electron group  (GLACIER; joint_model.py)
    - "add"  = adduct-shift group only              (ICEBERG/MARASON; dag_data)
  and the EVAL SET (which specs/CEs) — taken from that model's pred_eval.yaml so N
  matches exactly and the numbers are directly comparable.

USAGE
-----
  python analysis/magma_upper_bound.py \
     --pred-eval-yaml results/<model>_<dataset>/<run>/preds*/pred_eval.yaml \
     --dataset <dataset> [--convention both|add|all] [--max-peaks 100 20]

The pred_eval.yaml only supplies the (spec, CE) list the model was scored on; any
fragment model's yaml on the same fold yields the same oracle. `--dataset` defaults
--magma-tree to data/spec_datasets/<dataset>/magma_outputs/magma_tree.hdf5 and --gold
to data/spec_datasets/<dataset>/subformulae/no_subform.hdf5 (override either).

Reproduces the NIST'23 pilot-val numbers: convention "both" cos@100 0.8523 (cov 0.615,
72.1% intensity covered) / "add" cos@100 0.8062 (cov 0.550, 66.0% intensity).
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import ms_pred.common as common
import ms_pred.common.chem_utils as chem_utils


def get_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pred-eval-yaml", required=True,
                   help="a model's pred_eval.yaml; its 'name:' entries define the "
                        "exact (spec, CE) set to score the oracle on")
    p.add_argument("--dataset", default="nist23",
                   help="used to default --magma-tree and --gold paths")
    p.add_argument("--magma-tree", default=None,
                   help="fragment-set PredSpecDB (default: "
                        "data/spec_datasets/<dataset>/magma_outputs/magma_tree.hdf5). "
                        "Point this at a model's stage-1 tree_preds.hdf5 to get that "
                        "model's OWN stage-1 upper bound (the predicted DAG) instead of "
                        "the true-MAGMa bound.")
    p.add_argument("--dag-name-prefix", default="",
                   help="prefix the DAG stores on spec names relative to the eval yaml "
                        "(e.g. 'pred_' for ICEBERG stage-1 tree_preds.hdf5)")
    p.add_argument("--gold", default=None,
                   help="full true spectra (default: "
                        "data/spec_datasets/<dataset>/subformulae/no_subform.hdf5)")
    p.add_argument("--convention", choices=["both", "add", "all"], default="all",
                   help="reachable-set: both=GLACIER, add=ICEBERG/MARASON, all=report both")
    p.add_argument("--max-peaks", type=int, nargs="+", default=[100, 20])
    p.add_argument("--num-bins", type=int, default=15000)
    p.add_argument("--upper-limit", type=float, default=1500.0)
    p.add_argument("--max-broken", type=int, default=6,
                   help="max |hydrogen shift| considered when a fragment lacks explicit "
                        "max_add_hs/max_remove_hs (fallback only)")
    p.add_argument("--out", default=None, help="write results JSON here")
    return p.parse_args()


def main():
    a = get_args()
    data_dir = Path("data/spec_datasets") / a.dataset
    magma_tree = a.magma_tree or str(data_dir / "magma_outputs/magma_tree.hdf5")
    gold = a.gold or str(data_dir / "subformulae/no_subform.hdf5")
    NB, UP = a.num_bins, a.upper_limit
    SCALE = (NB - 1) / UP
    BUCKETS = np.linspace(0, UP, NB)
    massH = chem_utils.ELEMENT_TO_MASS["H"]
    SHIFTS = np.arange(-a.max_broken, a.max_broken + 1)
    conventions = ["both", "add"] if a.convention == "all" else [a.convention]

    def bin_gold(mz, inten):
        idx = np.floor(np.asarray(mz) * SCALE).astype(np.int32) + 1
        v = (idx >= 0) & (idx < NB)
        out = np.zeros(NB, dtype=np.float64)
        np.maximum.at(out, idx[v], np.asarray(inten, dtype=np.float64)[v])
        return out

    def reachable_masks(ms):
        base = np.asarray(ms.masses_no_adduct, dtype=float)
        max_add = np.asarray(ms.max_add_hs, dtype=int)
        max_rem = np.asarray(ms.max_remove_hs, dtype=int)
        ion = chem_utils.ion2mass[ms.adduct]
        e = (-chem_utils.ELECTRON_MASS if chem_utils.is_positive_adduct(ms.adduct)
             else chem_utils.ELECTRON_MASS)
        m_add, m_both = [], []
        for f in range(len(base)):
            s = SHIFTS[(SHIFTS >= -max_rem[f]) & (SHIFTS <= max_add[f])] * massH
            adduct_grp = base[f] + ion + s
            electron_grp = base[f] + e + s
            m_add.append(adduct_grp)
            m_both.append(adduct_grp); m_both.append(electron_grp)
        masks = {}
        for tag, groups in [("add", m_add), ("both", m_both)]:
            mask = np.zeros(NB, dtype=bool)
            if groups:
                mm = np.concatenate(groups); mm = mm[mm > 0]
                idx = np.clip(np.searchsorted(BUCKETS, mm, side="left"), 0, NB - 1)
                mask[idx] = True
            masks[tag] = mask
        return masks

    def score(true_vals, oracle_vals, k, true_norm):
        pos_cand = np.where(oracle_vals > 0)[0]
        order = pos_cand[np.argsort(-oracle_vals[pos_cand], kind="stable")][:k]
        pred = np.zeros_like(oracle_vals)
        pred[order] = oracle_vals[order]
        cos = float(np.dot(pred, true_vals) /
                    (max(np.linalg.norm(pred), 1e-6) * max(true_norm, 1e-6)))
        true_top = set(np.argsort(-true_vals, kind="stable")[:k].tolist())
        cov = len(true_top.intersection(order.tolist())) / max(len(true_top), 1e-6)
        return cos, cov

    # (spec, CE) set from the model's eval yaml
    names = []
    with open(a.pred_eval_yaml) as fp:
        for line in fp:
            if line.startswith("  name:"):
                names.append(line.split("name:", 1)[1].strip())
    spec_to_entries = defaultdict(list)
    for n in names:
        spec_to_entries[common.rm_collision_str(n)].append(n)
    print(f"eval entries: {len(names)}  unique specs: {len(spec_to_entries)}")

    mdb = common.PredSpecDB(magma_tree, mode="r")
    magma_names = set(mdb.get_all_names())
    gold_h5 = common.HDF5Dataset(gold)

    keys = [f"{m}{tag}" for tag in conventions
            for m in ([f"cos{k}_" for k in a.max_peaks] + [f"cov{k}_" for k in a.max_peaks]
                      + ["fpeak_", "finten_"])]
    acc = {k: [] for k in keys}
    miss_magma = miss_gold = n = 0
    t0 = time.time()

    for spec, entries in spec_to_entries.items():
        dag_spec = a.dag_name_prefix + spec
        if dag_spec not in magma_names:
            miss_magma += len(entries); continue
        ces, remarks = mdb.get_entries(dag_spec)
        ms = mdb.read(dag_spec, ces[0], remarks[0] if remarks else None)
        masks = reachable_masks(ms)
        for name in entries:
            gkey = f"{name}.json"
            if gkey not in gold_h5:
                miss_gold += 1; continue
            gj = json.loads(gold_h5.read_str(gkey))
            tbl = gj.get("output_tbl")
            if tbl is None:
                miss_gold += 1; continue
            inten_key = "ms2_inten" if "ms2_inten" in tbl else "rel_inten"
            true_spec = bin_gold(tbl["mono_mass"], tbl[inten_key])
            gb = np.where(true_spec > 0)[0]
            if len(gb) == 0:
                continue
            true_vals = true_spec[gb]
            true_norm = float(np.linalg.norm(true_vals))
            tot_inten = float(true_vals.sum())
            for tag in conventions:
                reach = masks[tag][gb]
                oracle_vals = np.where(reach, true_vals, 0.0)
                for k in a.max_peaks:
                    c, cov = score(true_vals, oracle_vals, k, true_norm)
                    acc[f"cos{k}_{tag}"].append(c)
                    acc[f"cov{k}_{tag}"].append(cov)
                acc[f"fpeak_{tag}"].append(float(reach.sum()) / len(gb))
                acc[f"finten_{tag}"].append(float(true_vals[reach].sum()) / max(tot_inten, 1e-12))
            n += 1
            if n % 10000 == 0:
                print(f"  ...{n} entries ({time.time()-t0:.0f}s)")
    gold_h5.close(); mdb.close()

    mean = lambda k: float(np.mean(acc[k])) if acc[k] else float("nan")
    print(f"\n=== MAGMa oracle  (N={n}, missing_magma={miss_magma}, missing_gold={miss_gold}) ===")
    label = {"both": "adduct+electron groups (GLACIER)", "add": "adduct group only (ICEBERG/MARASON)"}
    for tag in conventions:
        print(f"\n-- reachable set = {label[tag]} --")
        for k in a.max_peaks:
            print(f"  cos@{k} = {mean(f'cos{k}_{tag}'):.4f}   cov@{k} = {mean(f'cov{k}_{tag}'):.4f}")
        print(f"  frac true PEAKS covered     = {mean(f'fpeak_{tag}'):.4f}")
        print(f"  frac true INTENSITY covered = {mean(f'finten_{tag}'):.4f}")

    out = {k: mean(k) for k in acc}
    out.update(N=n, missing_magma=miss_magma, missing_gold=miss_gold,
               pred_eval_yaml=a.pred_eval_yaml, conventions=conventions)
    if a.out:
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
