"""GLACIER NIST'23 — predict test spectra and evaluate spectrum accuracy.

Uses version_0 checkpoints (no contrastive finetuning in this benchmark). Runs the
two benchmark splits and writes preds/pred_eval.yaml via analysis/spec_pred_eval.py.
"""
from pathlib import Path
import subprocess

python_file = "src/ms_pred/glacier/predict_inten_joint.py"
num_workers = 64
test_entries = [
    {"dataset": "nist23", "split": "split_1", "folder": "split_1_rnd1"},
    {"dataset": "nist23", "split": "scaffold_1", "folder": "scaffold_1_rnd1"},
]
devices = "0,1"

for test_entry in test_entries:
    split = test_entry["split"]
    dataset = test_entry["dataset"]
    folder = test_entry["folder"]

    res_folder = Path(f"results/glacier_{dataset}/")
    # version_0 = base joint model (no contrastive finetuning)
    model = res_folder / folder / "version_0/best.ckpt"
    if not model.exists():
        print(f"[skip] missing checkpoint: {model}")
        continue

    save_dir = model.parent.parent / "preds"

    cmd = f"""python {python_file} \\
    --batch-size {num_workers} \\
    --dataset-name {dataset} \\
    --split-name {split}.tsv \\
    --checkpoint {model} \\
    --save-dir {save_dir} \\
    --gpu \\
    --num-workers {num_workers} \\
    --subset-datasets test_only \\
    --binned-out"""
    cmd = f"CUDA_VISIBLE_DEVICES={devices} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True, check=True)

    out_binned = save_dir / "binned_preds.hdf5"
    eval_cmd = f"""python analysis/spec_pred_eval.py \\
    --binned-pred-file {out_binned} \\
    --max-peaks 100 \\
    --min-inten 0 \\
    --formula-dir-name no_subform.hdf5 \\
    --dataset {dataset}"""
    print(eval_cmd)
    subprocess.run(eval_cmd, shell=True, check=True)
