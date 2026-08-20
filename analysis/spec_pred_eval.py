""" Spectrum prediction evaluation

Use to compare binned predictions to ground truth spec values

"""
import ast
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
import yaml
import pickle
from collections import defaultdict
from functools import partial
from numpy.linalg import norm
from scipy.stats import sem

import ms_pred.common as common

DEFAULT_NUM_BINS = 15000
DEFAULT_UPPER_LIMIT = 1500


def cos_sim_fn(pred_ar, true_spec):
    """cos_sim_fn.

    Args:
        pred_ar:
        true_spec:
    """
    norm_pred = max(norm(pred_ar), 1e-6)
    norm_true = max(norm(true_spec), 1e-6)
    cos_sim = np.dot(pred_ar, true_spec) / (norm_pred * norm_true)
    return cos_sim


def entropy_sim_fn(pred_ar, true_spec):
    def norm_peaks(prob):
        return prob / (prob.sum(axis=-1, keepdims=True) + 1e-22)

    def entropy(prob):
        # assert np.all(np.abs(prob.sum(axis=-1) - 1) < 5e-3), f"Diff to 1: {np.max(np.abs(prob.sum(axis=-1) - 1))}"
        return -np.sum(prob * np.log(prob + 1e-22), axis=-1)

    norm_pred = norm_peaks(pred_ar)
    norm_true = norm_peaks(true_spec)
    entropy_pred = entropy(norm_pred)
    entropy_targ = entropy(norm_true)
    entropy_mix = entropy((norm_pred + norm_true) / 2)
    entropy_sim = 1 - (2 * entropy_mix - entropy_pred - entropy_targ) / np.log(4)
    return entropy_sim


def get_args():
    """get_args."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="nist20")
    parser.add_argument("--formula-dir-name", default="subform_20")
    parser.add_argument("--binned-pred-file")
    parser.add_argument("--outfile", default=None)
    parser.add_argument(
        "--min-inten",
        type=float,
        default=1e-5,
        help="Minimum intensity to call a peak in prediction",
    )
    parser.add_argument(
        "--max-peaks", type=int, default=20, help="Max num peaks to call"  # 20,
    )
    return parser.parse_args()


def process_spec_file(spec_name, num_bins: int, upper_limit: int, spec_dir: Path):
    """process_spec_file."""
    spec_h5 = common.HDF5Dataset(spec_dir)
    loaded_json = json.loads(spec_h5.read_str(f"{spec_name}.json"))

    if loaded_json.get("output_tbl") is None:
        return None

    # Load without adduct involved
    mz = loaded_json["output_tbl"]["mono_mass"]
    inten = loaded_json["output_tbl"]["ms2_inten"]
    spec_ar = np.vstack([mz, inten]).transpose(1, 0)
    binned = common.bin_spectra([spec_ar], num_bins, upper_limit)
    avged = binned[0]

    # normed = common.norm_spectrum(binned)
    # avged = normed.mean(0)
    return avged


def _get_spec_meta_value(spec_data, key: str):
    """Read metadata from either MassSpec attrs or legacy nested dict metadata."""
    if hasattr(spec_data, key):
        value = getattr(spec_data, key)
        if value is not None:
            return value

    if key == "num_bins" and hasattr(spec_data, "_num_bins"):
        value = getattr(spec_data, "_num_bins")
        if value is not None:
            return value

    if key == "upper_limit" and hasattr(spec_data, "_mass_upper_limit"):
        value = getattr(spec_data, "_mass_upper_limit")
        if value is not None:
            return value

    meta = getattr(spec_data, "meta", None)
    if isinstance(meta, dict) and key in meta:
        return meta[key]

    if isinstance(spec_data, dict):
        if key in spec_data:
            return spec_data[key]
        nested_meta = spec_data.get("meta")
        if isinstance(nested_meta, dict) and key in nested_meta:
            return nested_meta[key]

    defaults = {
        "num_bins": DEFAULT_NUM_BINS,
        "upper_limit": DEFAULT_UPPER_LIMIT,
    }
    if key in defaults:
        return defaults[key]

    raise KeyError(f"Could not find metadata field '{key}' in prediction entry")


def _load_predictions(binned_pred_file, data_df):
    """Load binned predictions, auto-detecting the writer's format.

    Supports three prediction layouts that coexist in this repo:
      * PredSpecDB HDF5 — fragment models (ICEBERG / SCARF / MARASON / GLACIER);
      * legacy binned HDF5 — pred_<spec>/ikey <k>/collision <ce>/spec, with
        per-collision `smiles`/`spec_name` attrs and file-level num_bins/upper_limit
        (MassFormer, GrAFF-MS);
      * pickle `.p` — dict of {preds, spec_names, smiles, num_bins, upper_limit}
        for collision-agnostic baselines (3DMolMS); each prediction is compared
        against every annotated collision energy of that spectrum.

    Returns (pred_spec_ars, pred_smiles, pred_spec_names, collision_energies,
    ion_types, num_bins, upper_limit).
    """
    binned_pred_file = Path(binned_pred_file)
    name_to_ion = dict(data_df[["spec", "ionization"]].values)

    pred_spec_ars, pred_smiles, pred_spec_names = [], [], []
    collision_energies, ion_types = [], []
    num_bins = upper_limit = None

    # --- pickle (collision-agnostic baselines, e.g. 3DMolMS) ---
    if binned_pred_file.suffix == ".p":
        with open(binned_pred_file, "rb") as fp:
            obj = pickle.load(fp)
        num_bins = int(obj["num_bins"])
        upper_limit = int(obj["upper_limit"])
        name_to_ces = {}
        for spec, ces in data_df[["spec", "collision_energies"]].values:
            try:
                name_to_ces[spec] = list(ast.literal_eval(ces)) if isinstance(ces, str) else list(ces)
            except (ValueError, SyntaxError):
                name_to_ces[spec] = []
        for pred, name, smi in zip(obj["preds"], obj["spec_names"], obj["smiles"]):
            name = common.rm_collision_str(name)
            for ce in name_to_ces.get(name, []) or [float("nan")]:
                pred_spec_ars.append(pred)
                pred_smiles.append(smi)
                pred_spec_names.append(name + f"_collision {float(ce):.0f}")
                collision_energies.append(float(ce))
                ion_types.append(name_to_ion.get(name))
        return (pred_spec_ars, pred_smiles, pred_spec_names, collision_energies,
                ion_types, num_bins, upper_limit)

    # --- HDF5: try PredSpecDB (fragment models) first ---
    try:
        pred_specs = common.PredSpecDB(h5_path=binned_pred_file, mode="r")
        for spec_tuple in pred_specs.get_all_specs():
            # get_all_specs yields (name, composite) or, when a per-spec remark is
            # stored (e.g. MassFormer writes the inchikey), (name, remark, composite).
            spec_id, spec_data2 = spec_tuple[0], spec_tuple[-1]
            for _spec_id2, spec_data in spec_data2.items():
                cur_upper = int(_get_spec_meta_value(spec_data, "upper_limit"))
                cur_bins = int(_get_spec_meta_value(spec_data, "num_bins"))
                if upper_limit is None:
                    upper_limit, num_bins = cur_upper, cur_bins
                name = common.rm_collision_str(spec_id)
                # Binned models (MassFormer) group CEs under one base-name composite,
                # so the CE lives on the per-CE MassSpec, not in spec_id; read it from
                # the spec and only fall back to parsing spec_id (fragment layout).
                ce = getattr(spec_data, "collision_energy", None)
                if ce is None:
                    ce = common.get_collision_energy(spec_id)
                # Fragment models store binned intensities under "intens"; binned
                # models (MassFormer) store them under "binned_spec".
                if getattr(spec_data, "has_intens", False):
                    binned = spec_data["intens"][:]
                else:
                    binned = spec_data["binned_spec"][:]
                pred_spec_ars.append(binned)
                pred_smiles.append(spec_data["root_canonical_smiles"])
                pred_spec_names.append(name + f"_collision {float(ce):.0f}")
                collision_energies.append(ce)
                ion_types.append(name_to_ion.get(name))
    except Exception:
        num_bins = upper_limit = None
        pred_spec_ars, pred_smiles, pred_spec_names = [], [], []
        collision_energies, ion_types = [], []

    if num_bins is not None:
        return (pred_spec_ars, pred_smiles, pred_spec_names, collision_energies,
                ion_types, num_bins, upper_limit)

    # --- HDF5: legacy binned layout (MassFormer / GrAFF-MS) ---
    pred_specs = common.HDF5Dataset(binned_pred_file)
    upper_limit = int(pred_specs.attrs["upper_limit"])
    num_bins = int(pred_specs.attrs["num_bins"])
    for pred_spec_obj in pred_specs.h5_obj.values():
        for smiles_obj in pred_spec_obj.values():
            name = smiles = None
            for ce_key, ce_obj in smiles_obj.items():
                if name is None:
                    name = ce_obj.attrs["spec_name"]
                if smiles is None:
                    smiles = ce_obj.attrs["smiles"]
                name = common.rm_collision_str(name)
                ce = common.get_collision_energy(ce_key)
                pred_spec_ars.append(ce_obj["spec"][:])
                pred_smiles.append(smiles)
                pred_spec_names.append(name + f"_collision {float(ce):.0f}")
                collision_energies.append(ce)
                ion_types.append(name_to_ion.get(name))
    pred_specs.close()
    return (pred_spec_ars, pred_smiles, pred_spec_names, collision_energies,
            ion_types, num_bins, upper_limit)


def main(args):
    """main."""
    dataset = args.dataset
    formula_dir_name = args.formula_dir_name
    data_folder = Path(f"data/spec_datasets/{dataset}/subformulae/{formula_dir_name}")
    min_inten = args.min_inten
    max_peaks = args.max_peaks
    data_label = data_folder.parent.parent / 'labels.tsv'
    data_df = pd.read_csv(data_label, sep='\t')
    name_to_ion = dict(data_df[["spec", "ionization"]].values)

    binned_pred_file = Path(args.binned_pred_file)
    outfile = Path(args.outfile) if args.outfile is not None else (
        binned_pred_file.parent / "pred_eval.yaml")
    outfile_grouped_template = str(outfile.parent / "pred_eval_grouped_{}.tsv")

    (
        pred_spec_ars,
        pred_smiles,
        pred_spec_names,
        collision_energies,
        ion_types,
        num_bins,
        upper_limit,
    ) = _load_predictions(binned_pred_file, data_df)

    if num_bins is None or upper_limit is None:
        raise ValueError(f"No prediction spectra found in {binned_pred_file}")

    read_spec = partial(
        process_spec_file,
        num_bins=num_bins,
        upper_limit=upper_limit,
        spec_dir=data_folder,
    )
    true_specs = common.chunked_parallel(
        pred_spec_names, read_spec, chunks=100, max_cpu=16
    )
    running_lists = defaultdict(lambda: [])
    output_entries = []
    for pred_ar, pred_smi, pred_spec, collision_energy, ion_type, true_spec in zip(
        pred_spec_ars, pred_smiles, pred_spec_names, collision_energies, ion_types, true_specs
    ):
        if true_spec is None:
            continue

        # Don't norm spec
        ## Norm pred spec by max
        # if np.max(pred_ar) > 0:
        #    pred_ar = np.array(pred_ar) / np.max(pred_ar)

        # Get all actual bins
        pred_greater = np.argwhere(pred_ar > min_inten).flatten()
        pos_bins_sorted = sorted(pred_greater, key=lambda x: -pred_ar[x])
        pos_bins = pos_bins_sorted[:max_peaks]

        new_pred = np.zeros_like(pred_ar)
        new_pred[pos_bins] = pred_ar[pos_bins]
        pred_ar = new_pred

        cos_sim = cos_sim_fn(pred_ar, true_spec)
        mse = np.mean((pred_ar - true_spec) ** 2)
        entropy_sim = entropy_sim_fn(pred_ar, true_spec)

        # Compute validity
        # Get all possible bins that would be valid
        if pred_smi is not None:
            true_form = common.form_from_smi(pred_smi)
            _cross_prod, masses = common.get_all_subsets(true_form)
            possible = common.digitize_ar(
                masses, num_bins=num_bins, upper_limit=upper_limit
            )
            smiles_mass = common.mass_from_smi(pred_smi)
            ikey = common.inchikey_from_smiles(pred_smi)

            max_possible_bin = np.max(possible)

        else:
            possible = []
            smiles_mass = 0
            max_possible_bin = 0
            ikey = ""

        copy_pred, copy_true = np.copy(pred_ar), np.copy(true_spec)
        copy_pred[max_possible_bin] = 0
        copy_true[max_possible_bin] = 0
        cos_sim_zero_pep = cos_sim_fn(copy_pred, copy_true)

        # TODO computing validity should use masses with adduct and consider adduct transfer
        # possible_set = set(possible)
        # pred_set = set(pos_bins)
        # overlap = pred_set.intersection(possible_set)
        #
        # # Check invalid smiles here
        # invalid = pred_set.difference(possible_set)
        #
        # if len(pred_set) == 0:
        #     frac_valid = 1.0
        # else:
        #     frac_valid = len(overlap) / len(pred_set)

        # Compute true overlap
        true_inds = np.argwhere(true_spec > min_inten).flatten()
        true_bins_sorted = sorted(true_inds, key=lambda x: -true_spec[x])
        true_bins = set(true_bins_sorted[:max_peaks])

        total_covered = true_bins.intersection(pos_bins)
        overlap_coeff = len(total_covered) / max(
            min(len(true_bins), len(pos_bins)), 1e-6
        )
        coverage = len(total_covered) / max(len(true_bins), 1e-6)

        # Load true spec
        output_entry = {
            "name": str(pred_spec),
            "inchi": ikey,
            "cos_sim": float(cos_sim),
            "cos_sim_zero_pep": float(cos_sim_zero_pep),
            "mse": float(mse),
            "entropy_sim": float(entropy_sim),
            # "frac_valid": float(frac_valid),
            "overlap_coeff": float(overlap_coeff),
            "coverage": float(coverage),
            "len_targ": len(true_bins),
            "len_pred": len(pos_bins),
            "compound_mass": smiles_mass,
            "mass_bin": common.bin_mass_results(smiles_mass),
            "collision_bin": common.bin_collision_results(collision_energy),
            "ion_type": ion_type,
        }

        output_entries.append(output_entry)
        running_lists["cos_sim"].append(cos_sim)
        running_lists["cos_sim_zero_pep"].append(cos_sim_zero_pep)
        running_lists["mse"].append(mse)
        running_lists["entropy_sim"].append(entropy_sim)
        # running_lists["frac_valid"].append(frac_valid)
        running_lists["overlap_coeff"].append(overlap_coeff)
        running_lists["coverage"].append(coverage)
        running_lists["len_targ"].append(len(true_bins))
        running_lists["len_pred"].append(len(pos_bins))

    final_output = {
        "dataset": dataset,
        "data_folder": str(data_folder),
        "individuals": sorted(output_entries, key=lambda x: x["cos_sim"]),
    }

    for k, v in running_lists.items():
        final_output[f"avg_{k}"] = float(np.mean(v))
        # Add std and sem
        final_output[f"std_{k}"] = float(np.std(v))
        final_output[f"sem_{k}"] = float(sem(v))

    df = pd.DataFrame(output_entries)
    for grouped_key in ["mass_bin", "collision_bin", "ion_type"]:
        df_grouped = pd.concat(
            [df.groupby(grouped_key).mean(numeric_only=True), df.groupby(grouped_key).size()], axis=1
        )
        df_grouped = df_grouped.rename({0: "num_examples"}, axis=1)

        all_mean = df.mean(numeric_only=True)
        all_mean["num_examples"] = len(df)
        all_mean.name = "avg"
        df_grouped = pd.concat([df_grouped, all_mean.to_frame().T], axis=0)
        df_grouped.to_csv(outfile_grouped_template.format(grouped_key), sep="\t")

    with open(outfile, "w") as fp:
        out_str = yaml.dump(final_output, indent=2)
        print(out_str)
        fp.write(out_str)


if __name__ == "__main__":
    """__main__"""
    args = get_args()
    main(args)
