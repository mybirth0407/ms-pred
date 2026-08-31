"""SCARF NIST'23 — predict test spectra and evaluate spectrum accuracy.

Runs the scaffold split (scaffold_1, seed 1) and writes
preds/pred_eval.yaml via analysis/spec_pred_eval.py.
"""
from pathlib import Path
import subprocess

python_file = "src/ms_pred/scarf_pred/predict_inten.py"
devices = "0"
node_num = 300
test_entries = [
    {"dataset": "nist23", "split": "scaffold_1", "folder": "scaffold_1_rnd1"},
]

for test_entry in test_entries:
    dataset = test_entry["dataset"]
    split = test_entry["split"]
    folder = test_entry["folder"]

    res_folder = Path(f"results/scarf_inten_{dataset}/")
    model = res_folder / folder / "version_0/best.ckpt"
    if not model.exists():
        print(f"[skip] missing checkpoint: {model}")
        continue

    base_formula_folder = Path(f"results/scarf_{dataset}")
    save_dir = model.parent.parent / "preds"
    formula_folder = base_formula_folder / folder / f"preds_train_{node_num}/form_preds"

    cmd = f"""python {python_file} \\
    --batch-size 32 \\
    --dataset-name {dataset} \\
    --split-name {split}.tsv \\
    --checkpoint {model} \\
    --save-dir {save_dir} \\
    --gpu \\
    --num-workers 0 \\
    --subset-datasets test_only \\
    --formula-folder {formula_folder} \\
    --binned-out"""
    cmd = f"CUDA_VISIBLE_DEVICES={devices} {cmd}"
    print(cmd + "\n")
    subprocess.run(cmd, shell=True, check=True)

    out_binned = save_dir / "binned_preds.p"
    eval_cmd = f"""python analysis/spec_pred_eval.py \\
    --binned-pred-file {out_binned} \\
    --max-peaks 100 \\
    --min-inten 0 \\
    --formula-dir-name no_subform \\
    --dataset {dataset}"""
    print(eval_cmd)
    subprocess.run(eval_cmd, shell=True, check=True)
