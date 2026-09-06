#!/bin/bash
#SBATCH --job-name=pmcs-B-150-hd95
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa4000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=pmcs-splitB-150-hd95-%j.out
#SBATCH --error=pmcs-splitB-150-hd95-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pmcs
cd "$HOME/pmcs"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

DATA_SEED=20260905
MODEL_SEED=42
SPLIT_CONFIG="split/generated/data_seed_${DATA_SEED}/splitB/materialized_config.json"
VERSION="pmcs_splitB_150r_hd95_seed${MODEL_SEED}_data${DATA_SEED}_job${SLURM_JOB_ID}"
RESULT_DIR="results/${VERSION}"

test -f "$SPLIT_CONFIG"

python -m py_compile \
  options.py train_federated.py dataset/datasets.py utils/metrics.py \
  scripts/audit_full_run.py

python - <<'PY'
import numpy as np
from utils.metrics import hd95_case

empty = np.zeros((240, 240, 155), dtype=np.uint8)
prediction = empty.copy()
prediction[0, 0, 0] = 1
values, tags = hd95_case(prediction, empty)
assert abs(values['WT'] - 373.1286641361127) < 1e-6
assert tags['WT'] == 'mismatch_empty'
print('HD95 PRE-FLIGHT PASSED')
PY

python -u train_federated.py \
  --client_num 8 \
  --gpus 0,1,2,3 \
  --batch_size 1 \
  --local_ep 1 \
  --c_rounds 150 \
  --eval 10 \
  --datapath "$HOME/FedMEPD/data/BraTS2020_npy" \
  --dataname BRATS2020 \
  --split_config "$SPLIT_CONFIG" \
  --data_seed "$DATA_SEED" \
  --seed "$MODEL_SEED" \
  --lam_sd 0.1 \
  --compute_hd95 \
  --compute_pers_gain \
  --eval_global_model \
  --version "$VERSION" \
  --resume 0

python scripts/audit_full_run.py "$RESULT_DIR" \
  --split B --rounds 150 --eval_every 10

echo "SPLIT B 150-ROUND TRAINING COMPLETED"
