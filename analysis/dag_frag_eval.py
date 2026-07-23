"""Stage-1 fragment-DAG quality evaluation (HDF5 / MassSpec aware).

Modern replacement for ``analysis/dag_pred_eval.py`` (which globs obsolete
``*.json`` trees).  It compares a *predicted* fragment DAG produced by a 2-stage
fragment model's generative stage (ICEBERG / MARASON ``predict_gen.py`` ->
``tree_preds.hdf5``) against the *gold* MAGMa DAG
(``.../magma_outputs/magma_tree.hdf5``), and reports set-overlap quality of the
generated fragment nodes:

    recall (= coverage) = |pred & true| / |true|
    precision           = |pred & true| / |pred|
    f1                  = 2 P R / (P + R)          (0 if P + R == 0)
    jaccard             = |pred & true| / |pred | true|

FRAGMENT IDENTITY
-----------------
Both the gold tree and the predicted tree are stored as
``ms_pred.common.PredSpecDB`` / ``MassSpec`` objects.  A MassSpec exposes a
fragment set through related views:

  * ``.frags``          -> (n_frag, n_atom) bool bit-vectors (atom subset kept)
  * ``.int_frags``      -> the same as integer atom bitmasks
  * ``.frag_form``      -> per-fragment molecular formula string
  * ``.frag_form_vecs`` -> the dense element vector behind ``.frag_form``

The original JSON-era evaluator intersected ``tree["frags"].keys()``, and those
keys were the MAGMa *WL graph hashes* (``FragmentEngine.wl_hash(frag)`` — see
``magma/run_magma.py`` where ``out_frags`` is keyed by ``frag_hash`` and
``magma/fragmentation.py`` where DAG nodes are de-duplicated by ``wl_hash``).

The raw atom bitmask (``.int_frags`` / ``.frags``) is NOT usable to intersect a
gold tree against a predicted tree: ``iceberg/predict_gen.py`` canonicalises the
molecule with ``common.rm_stereo`` before fragmenting, so the predicted root
SMILES loses the stereochemistry that the gold tree keeps.  The two canonical
SMILES therefore differ, RDKit assigns different atom indices, and identical
substructures get different bitmasks (empirically recall collapses to ~0.5).

``wl_hash`` is computed from the fragment's labelled graph (atom symbols, H
counts, bond types) and is invariant to atom re-indexing and stereochemistry, so
it matches fragments correctly across the two pipelines *and* reproduces the
original JSON key semantics.  It is therefore the default identity here.
``--identity formula`` is offered as a lighter, dependency-free alternative (no
RDKit / FragmentEngine); it is coarser because it merges constitutional isomers
that share a molecular formula.  ``--identity atoms`` is kept only for diagnostics
(it is the un-aligned raw bitmask described above and will under-count).
"""
import argparse
import functools
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import sem

import ms_pred.common as common
import ms_pred.magma.fragmentation as fragmentation

DEFAULT_GOLD = "data/spec_datasets/nist23/magma_outputs/magma_tree.hdf5"

# Fields to summarise (avg / sem / std) and to write into the grouped TSVs.
_METRIC_KEYS = ["precision", "recall", "f1", "jaccard", "num_pred", "num_true", "intersect"]


def get_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pred-tree-h5", required=True,
                        help="Predicted tree_preds.hdf5 (PredSpecDB) from predict_gen.py")
    parser.add_argument("--gold-tree-h5", default=DEFAULT_GOLD,
                        help="Gold MAGMa magma_tree.hdf5 (PredSpecDB)")
    parser.add_argument("--dataset", default="nist23")
    parser.add_argument("--outfile", default=None,
                        help="YAML output path (default: <pred-tree parent>/pred_eval_frag.yaml)")
    parser.add_argument("--identity", default="wl", choices=["wl", "formula", "atoms"],
                        help="Fragment identity for set intersection (default: wl)")
    parser.add_argument("--pred-prefix", default="pred_",
                        help="Prefix stripped from predicted spec ids to match gold ids")
    parser.add_argument("--limit", default=None, type=int,
                        help="Only evaluate the first N predicted spec ids (for quick CPU validation)")
    parser.add_argument("--max-cpu", default=1, type=int,
                        help="Parallel workers over predicted spec ids (default 1 = sequential)")
    parser.add_argument("--split-name", default=None,
                        help="If set, restrict scoring to spec ids in this split's subset "
                             "(e.g. scaffold_1.tsv) so a whole-dataset tree_preds is scored test-only")
    parser.add_argument("--subset", default="test",
                        help="Split subset to keep when --split-name is given (default: test)")
    parser.add_argument("--dataset-dir", default=None,
                        help="Dataset dir holding splits/ (default: data/spec_datasets/<dataset>)")
    return parser.parse_args()


def _int_frag_from_row(row, natoms=None):
    """Atom bitmask integer from a bool bit-vector row (padding-safe)."""
    nz = np.flatnonzero(np.asarray(row, dtype=bool))
    if natoms is not None:
        nz = nz[nz < natoms]
    val = 0
    for i in nz.tolist():
        val |= (1 << int(i))
    return val


class _EngineCache:
    """Cache FragmentEngine instances by (canonical) SMILES within a process."""

    def __init__(self):
        self._cache = {}

    def get(self, smi):
        eng = self._cache.get(smi)
        if eng is None:
            eng = fragmentation.FragmentEngine(
                mol_str=smi, mol_str_type="smiles", mol_str_canonicalized=True
            )
            self._cache[smi] = eng
        return eng


def frag_identities(ms, identity, engine_cache):
    """Return a set of hashable fragment identities for a MassSpec."""
    if ms is None or not getattr(ms, "has_frags", False) or ms.frags is None or len(ms.frags) == 0:
        return set()

    if identity == "formula":
        return set(ms.frag_form)

    if identity == "atoms":
        return set(tuple(np.flatnonzero(np.asarray(r, dtype=bool)).tolist()) for r in ms.frags)

    if identity == "wl":
        smi = ms.root_canonical_smiles
        eng = engine_cache.get(smi)
        natoms = eng.natoms
        return set(eng.wl_hash(_int_frag_from_row(r, natoms)) for r in ms.frags)

    raise ValueError(f"Unknown identity {identity}")


def _std_ce(ce):
    """Standardise a collision energy to the PredSpecDB string key (e.g. '40')."""
    try:
        return f"{float(ce):.0f}"
    except (TypeError, ValueError):
        return str(ce)


def _entry_for_pair(pred_ms, gold_ms, identity, engine_cache, name, ce, gold_name):
    """Compute one output record from a matched (pred, gold) MassSpec pair."""
    true_ids = frag_identities(gold_ms, identity, engine_cache)
    pred_ids = frag_identities(pred_ms, identity, engine_cache)
    num_true = len(true_ids)
    num_pred = len(pred_ids)
    if num_true == 0:
        return None  # nothing to score against

    intersect = len(true_ids & pred_ids)
    union = len(true_ids | pred_ids)
    recall = intersect / num_true
    precision = intersect / num_pred if num_pred > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    jaccard = intersect / union if union > 0 else 0.0

    smi = gold_ms.root_canonical_smiles
    try:
        compound_mass = common.mass_from_smi(smi)
    except Exception:
        compound_mass = float("nan")
    try:
        ikey = common.inchikey_from_smiles(smi)
    except Exception:
        ikey = ""

    ce_str = _std_ce(ce)
    ce_val = float(ce_str) if ce_str != "nan" else float("nan")
    return {
        "name": f"{gold_name}_collision {ce_str}",
        "spec": gold_name,
        "collision_energy": ce_val,
        "inchikey": ikey,
        "num_pred": num_pred,
        "num_true": num_true,
        "intersect": intersect,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "compound_mass": compound_mass,
        "mass_bin": common.bin_mass_results(compound_mass),
        "collision_bin": common.bin_collision_results(ce_val) if ce_str != "nan" else None,
    }


def _process_names(pred_names, pred_db, gold_db, identity, pred_prefix, engine_cache):
    """Score a list of predicted spec ids. Returns (entries, n_missing_gold)."""
    entries = []
    n_missing = 0
    for pn in pred_names:
        gold_name = pn[len(pred_prefix):] if pn.startswith(pred_prefix) else pn
        try:
            pred_ces, pred_rems = pred_db.get_entries(pn)
        except Exception:
            continue
        try:
            gold_ces, gold_rems = gold_db.get_entries(gold_name)
        except Exception:
            gold_ces, gold_rems = [], []
        gold_ce_to_rem = {}
        for gce, grem in zip(gold_ces, gold_rems):
            gold_ce_to_rem.setdefault(_std_ce(gce), grem)

        for ce, rem in zip(pred_ces, pred_rems):
            key = _std_ce(ce)
            if key not in gold_ce_to_rem:
                n_missing += 1
                continue
            try:
                pred_ms = pred_db.read(pn, ce, rem)
                gold_ms = gold_db.read(gold_name, key, gold_ce_to_rem[key])
            except Exception:
                n_missing += 1
                continue
            rec = _entry_for_pair(pred_ms, gold_ms, identity, engine_cache, pn, ce, gold_name)
            if rec is not None:
                entries.append(rec)
    return entries, n_missing


# ----- parallel worker plumbing -----
# Workers may be launched with the 'spawn' start method (fresh module import),
# so all config is passed as arguments (bound via functools.partial). DB handles
# are cached per-process in _DB_CACHE keyed by (pred_path, gold_path) so repeated
# chunk calls in the same worker reuse the open files.
_DB_CACHE = {}


def _worker(pred_path, gold_path, identity, pred_prefix, pred_names):
    key = (pred_path, gold_path)
    cached = _DB_CACHE.get(key)
    if cached is None:
        cached = (
            common.PredSpecDB(pred_path, mode="r"),
            common.PredSpecDB(gold_path, mode="r"),
            _EngineCache(),
        )
        _DB_CACHE[key] = cached
    pred_db, gold_db, engine_cache = cached
    return _process_names(pred_names, pred_db, gold_db, identity, pred_prefix, engine_cache)


def main(args):
    pred_path = Path(args.pred_tree_h5)
    gold_path = Path(args.gold_tree_h5)
    identity = args.identity

    outfile = Path(args.outfile) if args.outfile else pred_path.parent / "pred_eval_frag.yaml"
    outfile_grouped_template = str(outfile.parent / "pred_eval_frag_grouped_{}.tsv")

    pred_db = common.PredSpecDB(str(pred_path), mode="r")
    pred_names = [n for n in pred_db.get_all_names()]
    if args.split_name is not None:
        import pandas as pd
        ddir = Path(args.dataset_dir) if args.dataset_dir else Path(f"data/spec_datasets/{args.dataset}")
        sdf = pd.read_csv(ddir / "splits" / args.split_name, sep="\t")
        id_col, split_col = sdf.columns[0], sdf.columns[1]
        keep = set(sdf[sdf[split_col] == args.subset][id_col])

        def _spec_of(n):
            s = n[len(args.pred_prefix):] if n.startswith(args.pred_prefix) else n
            return common.rm_collision_str(s)

        before = len(pred_names)
        pred_names = [n for n in pred_names if _spec_of(n) in keep]
        print(f"Restricted to {args.subset} of {args.split_name}: {len(pred_names)}/{before} spec ids")
    if args.limit is not None:
        pred_names = pred_names[: args.limit]
    print(f"Evaluating {len(pred_names)} predicted spec ids from {pred_path}")
    print(f"Gold trees: {gold_path}  |  identity: {identity}")

    if args.max_cpu > 1:
        # Chunk names across workers; each worker opens its own DB handles.
        n_chunks = max(args.max_cpu * 4, 1)
        chunk_sz = max(1, (len(pred_names) + n_chunks - 1) // n_chunks)
        name_chunks = [pred_names[i:i + chunk_sz] for i in range(0, len(pred_names), chunk_sz)]
        pred_db.close()  # workers reopen their own
        worker_fn = functools.partial(
            _worker, str(pred_path), str(gold_path), identity, args.pred_prefix
        )
        results = common.chunked_parallel(
            name_chunks, worker_fn, chunks=len(name_chunks), max_cpu=args.max_cpu,
            task_name="dag_frag_eval",
        )
        output_entries = [e for chunk_res, _ in results for e in chunk_res]
        num_missing = sum(m for _, m in results)
    else:
        gold_db = common.PredSpecDB(str(gold_path), mode="r")
        engine_cache = _EngineCache()
        output_entries, num_missing = _process_names(
            pred_names, pred_db, gold_db, identity, args.pred_prefix, engine_cache
        )

    if len(output_entries) == 0:
        raise ValueError(
            f"No matched (spec, collision) pairs between {pred_path} and {gold_path}. "
            "Check --pred-prefix and that the gold tree covers these spec ids."
        )

    running = defaultdict(list)
    for e in output_entries:
        for k in _METRIC_KEYS:
            running[k].append(e[k])

    final_output = {
        "dataset": args.dataset,
        "pred_tree": str(pred_path),
        "gold_tree": str(gold_path),
        "identity": identity,
        "num_matched_entries": len(output_entries),
        "num_pred_specs": len(pred_names),
        "num_unmatched_collisions": int(num_missing),
        "individuals": sorted(output_entries, key=lambda x: x["f1"]),
    }
    for k, v in running.items():
        final_output[f"avg_{k}"] = float(np.mean(v))
        final_output[f"sem_{k}"] = float(sem(v)) if len(v) > 1 else 0.0
        final_output[f"std_{k}"] = float(np.std(v))

    # Grouped TSVs (mirror analysis/spec_pred_eval.py).
    df = pd.DataFrame(output_entries)
    for grouped_key in ["mass_bin", "collision_bin"]:
        df_grouped = pd.concat(
            [df.groupby(grouped_key).mean(numeric_only=True),
             df.groupby(grouped_key).size()], axis=1
        )
        df_grouped = df_grouped.rename({0: "num_examples"}, axis=1)
        all_mean = df.mean(numeric_only=True)
        all_mean["num_examples"] = len(df)
        all_mean.name = "avg"
        df_grouped = pd.concat([df_grouped, all_mean.to_frame().T], axis=0)
        df_grouped.to_csv(outfile_grouped_template.format(grouped_key), sep="\t")

    with open(outfile, "w") as fp:
        out_str = yaml.dump(final_output, indent=2, sort_keys=False)
        fp.write(out_str)

    # Console summary
    print(f"\n=== stage-1 fragment DAG eval ({identity}) ===")
    print(f"matched entries : {len(output_entries)}  (unmatched collisions: {num_missing})")
    for k in ["precision", "recall", "f1", "jaccard"]:
        print(f"avg_{k:<9}: {final_output[f'avg_{k}']:.4f}  (sem {final_output[f'sem_{k}']:.4f})")
    print(f"avg_num_true    : {final_output['avg_num_true']:.2f}")
    print(f"avg_num_pred    : {final_output['avg_num_pred']:.2f}")
    print(f"\nwrote: {outfile}")
    print(f"wrote: {outfile_grouped_template.format('mass_bin')}")
    print(f"wrote: {outfile_grouped_template.format('collision_bin')}")


if __name__ == "__main__":
    main(get_args())
