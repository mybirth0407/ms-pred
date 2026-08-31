"""MassFormer NIST'23 — predict test spectra and evaluate spectrum accuracy.

Runs the scaffold split (scaffold_1, seed 1) and
writes preds/pred_eval.yaml (avg_cos_sim / avg_entropy_sim / avg_coverage) via
analysis/spec_pred_eval.py.
"""
from pathlib import Path
import subprocess

python_file = "src/ms_pred/massformer_pred/predict.py"
devices = "0"
test_entries = [
    {"dataset": "nist23", "split": "scaffold_1", "folder": "scaffold_1_rnd1"},
]

for test_entry in test_entries:
    split = test_entry["split"]
    folder = test_entry["folder"]
    dataset_name = test_entry["dataset"]

    res_folder = Path(f"results/massformer_baseline_{dataset_name}")
    model = res_folder / f"{folder}/version_0/best.ckpt"
    if not model.exists():
        print(f"[skip] missing checkpoint: {model}")
        continue

    save_dir = model.parent.parent / "preds"
    save_dir.mkdir(exist_ok=True, parents=True)

    cmd = f"""python {python_file} \\
    --batch-size 32 \\
    --dataset-name {dataset_name} \\
    --split-name {split}.tsv \\
    --subset-datasets test_only  \\
    --checkpoint {model} \\
    --save-dir {save_dir} \\
    --gpu"""
    cmd = f"CUDA_VISIBLE_DEVICES={devices} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True, check=True)

    out_binned = save_dir / "binned_preds.hdf5"
    eval_cmd = f"""python analysis/spec_pred_eval.py \\
    --binned-pred-file {out_binned} \\
    --max-peaks 100 \\
    --min-inten 0 \\
    --formula-dir-name no_subform.hdf5 \\
    --dataset {dataset_name}"""
    print(eval_cmd)
    subprocess.run(eval_cmd, shell=True, check=True)
