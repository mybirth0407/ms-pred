. data_scripts/dag/run_magma.sh
. run_scripts/glacier/add_inten.sh
. run_scripts/glacier/01_train_joint.sh
python run_scripts/glacier/02_predict_inten.py
python run_scripts/glacier/03_run_retrieval.py