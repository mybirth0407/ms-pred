python launcher_scripts/run_from_config.py configs/iceberg/nist23/dag_inten_train_nist23.yaml

# Contrastive finetuning is omitted for the spectrum-only benchmark (it trains against
# PubChem decoys to improve retrieval ranking, out of scope here). The checkpoint used
# for evaluation is results/iceberg_nist23/<split>/ckpt/inten/best.ckpt. To enable it,
# set num-decoys: [0, 10] in dag_gen_predict_train_nist23.yaml, provide the PubChem map,
# and uncomment the line below.
# python launcher_scripts/run_from_config.py configs/iceberg/nist23/dag_inten_contr_finetune_nist23.yaml
