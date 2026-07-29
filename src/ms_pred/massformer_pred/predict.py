"""predict.py

Make predictions with trained model

"""

import logging
from datetime import datetime
import yaml
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

import ms_pred.common as common
import ms_pred.nn_utils as nn_utils
from ms_pred.massformer_pred import massformer_data, massformer_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--gpu", default=False, action="store_true")
    parser.add_argument("--sparse-out", default=False, action="store_true")
    parser.add_argument("--sparse-k", default=100, action="store", type=int)
    parser.add_argument(
        "--num-bins", default=-1, action="store", type=int,
        help="Binning resolution for stored spectra; -1 uses the model's output_dim.",
    )
    parser.add_argument(
        "--upper-limit", default=-1, action="store", type=int,
        help="Upper m/z limit for binning; -1 uses the model's upper_limit.",
    )
    parser.add_argument(
        "--out-name", default="binned_preds.hdf5", action="store",
        help="Output HDF5 filename written into --save-dir. Use e.g. "
             "binned_preds_shard0.hdf5 for sharded parallel prediction (PredSpecDB "
             "auto-merges sibling *_shard*.hdf5 files on read).",
    )
    parser.add_argument("--num-workers", default=0, action="store", type=int)
    parser.add_argument("--batch-size", default=64, action="store", type=int)
    date = datetime.now().strftime("%Y_%m_%d")
    parser.add_argument("--save-dir", default=f"results/{date}_massformer_pred/")
    parser.add_argument(
        "--checkpoint-pth",
        help="name of checkpoint file",
    )
    parser.add_argument("--dataset-name", default="gnps2015_debug")
    parser.add_argument("--dataset-labels", default="labels.tsv")
    parser.add_argument("--split-name", default="split_22.tsv")
    parser.add_argument(
        "--subset-datasets",
        default="none",
        action="store",
        choices=["none", "train_only", "test_only"],
    )
    return parser.parse_args()


def predict():
    args = get_args()
    kwargs = args.__dict__
    sparse_out = kwargs["sparse_out"]
    sparse_k = kwargs["sparse_k"]

    save_dir = kwargs["save_dir"]
    log_name = Path(kwargs["out_name"]).stem + ".log"
    common.setup_logger(save_dir, log_name=log_name, debug=kwargs["debug"])
    pl.seed_everything(kwargs.get("seed"))

    # Dump args
    yaml_args = yaml.dump(kwargs)
    logging.info(f"\n{yaml_args}")
    with open(Path(save_dir) / "args.yaml", "w") as fp:
        fp.write(yaml_args)

    # Get dataset
    # Load smiles dataset and split into 3 subsets
    dataset_name = kwargs["dataset_name"]
    data_dir = Path("data/spec_datasets") / dataset_name
    labels = data_dir / kwargs["dataset_labels"]

    # Get train, val, test inds
    df = pd.read_csv(labels, sep="\t")

    if kwargs["subset_datasets"] != "none":
        splits = pd.read_csv(data_dir / "splits" / kwargs["split_name"], sep="\t")
        folds = set(splits.keys())
        folds.remove("spec")
        fold_name = list(folds)[0]
        if kwargs["subset_datasets"] == "train_only":
            names = splits[splits[fold_name] == "train"]["spec"].tolist()
        elif kwargs["subset_datasets"] == "test_only":
            names = splits[splits[fold_name] == "test"]["spec"].tolist()
        else:
            raise NotImplementedError()
        df = df[df["spec"].isin(names)]

    num_workers = kwargs.get("num_workers", 0)

    # Create model and load
    # Load from checkpoint
    best_checkpoint = kwargs["checkpoint_pth"]
    model = massformer_model.MassFormer.load_from_checkpoint(best_checkpoint)
    logging.info(f"Loaded model with from {best_checkpoint}")
    pred_dataset = massformer_data.MolDataset(
        df,
        num_workers=num_workers,
    )

    # Define dataloaders
    collate_fn = pred_dataset.get_collate_fn()
    pred_loader = DataLoader(
        pred_dataset,
        num_workers=kwargs["num_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        batch_size=kwargs["batch_size"],
    )

    model.eval()
    device = "cuda" if kwargs["gpu"] else "cpu"
    device = torch.device(device)
    model = model.to(device)

    num_bins = kwargs["num_bins"] if kwargs["num_bins"] > 0 else model.output_dim
    upper_limit = kwargs["upper_limit"] if kwargs["upper_limit"] > 0 else model.upper_limit

    out_file = Path(kwargs["save_dir"]) / kwargs["out_name"]
    # Write via PredSpecDB (binned-only spectra) so downstream readers
    # (analysis/spec_pred_eval.py, retrieval/retrieval_benchmark.py) can consume
    # MassFormer predictions through the same code path as the fragment models.
    specdb = common.PredSpecDB(
        h5_path=out_file, mode="w",
        has_probs=False, has_brokens=False, has_masses=False, has_masses_no_adduct=False,
        has_frag_form_vecs=False, has_frags=False, has_intens=False, has_binned_spec=True,
    )
    with torch.no_grad():
        for batch in tqdm(pred_loader):
            graphs, smiles, weights, adducts, collision_energies, norm_collision_energies = (
                batch["gf_v2_data"],
                batch["names"],
                batch["full_weight"],
                batch["adducts"],
                batch["collision_energies"],
                batch["norm_collision_energies"]
            )

            spec_names = batch["spec_names"]

            graphs = nn_utils.dict_to_device(graphs, device)
            weights = weights.to(device)
            adducts = adducts.to(device)
            norm_collision_energies = norm_collision_energies.to(device)

            output = model.predict(graphs, weights, adducts, norm_collision_energies)
            output_spec = output["spec"].cpu().detach().numpy()

            # Shrink it to only top k, ordering inds, intens
            if sparse_out:
                best_inds = np.argsort(output_spec, -1)[:, ::-1][:, :sparse_k]
                best_intens = np.take_along_axis(output_spec, best_inds, -1)
                output_spec = np.stack([best_inds, best_intens], -1)

            for spec_name, smi, collision_energy, out_spec in zip(spec_names, smiles, collision_energies, output_spec):
                inchikey = common.inchikey_from_smiles(smi)
                ce = float(collision_energy)
                if sparse_out:
                    ms = common.MassSpec(
                        collision_energy=ce, root_canonical_smiles=smi, adduct=None, remark=inchikey,
                        binned_spec_sparse=(
                            out_spec[:, 0].astype(np.uint32), out_spec[:, 1].astype(np.float32),
                        ),
                        num_bins=num_bins, upper_limit=upper_limit,
                    )
                else:
                    ms = common.MassSpec(
                        collision_energy=ce, root_canonical_smiles=smi, adduct=None, remark=inchikey,
                        binned_spec=out_spec.astype(np.float32),
                        num_bins=num_bins, upper_limit=upper_limit,
                    )
                specdb.write(spec_name, ms)

    specdb.close()


if __name__ == "__main__":
    import time

    start_time = time.time()
    predict()
    end_time = time.time()
    logging.info(f"Program finished in: {end_time - start_time} seconds")
