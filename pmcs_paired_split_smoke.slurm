#!/bin/bash
#SBATCH --job-name=pmcs-split-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa4000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=pmcs-split-smoke-%j.out
#SBATCH --error=pmcs-split-smoke-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pmcs
cd "$HOME/pmcs"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

DATA_SEED=20260905
GENERATED_ROOT="split/generated/data_seed_${DATA_SEED}"
MAPPING="$HOME/FedMEPD/data/BraTS2020_raw/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/name_mapping.csv"
VOLUMES="$HOME/FedMEPD/data/BraTS2020_npy/vol"

python -m py_compile options.py train_federated.py dataset/datasets.py utils/metrics.py \
  scripts/validate_split.py scripts/materialize_splits.py scripts/audit_split_smoke.py

python scripts/validate_split.py \
  configs/splitA.json configs/splitB.json \
  --data_seed "$DATA_SEED" \
  --manifest "$MAPPING" \
  --volume_dir "$VOLUMES"

for SPLIT_ID in A B; do
  SPLIT_CONFIG="$GENERATED_ROOT/split${SPLIT_ID}/materialized_config.json"
  test -f "$SPLIT_CONFIG"

  python -u train_federated.py \
    --client_num 8 \
    --gpus 0,1,2,3 \
    --batch_size 1 \
    --local_ep 1 \
    --c_rounds 2 \
    --eval 1 \
    --datapath "$HOME/FedMEPD/data/BraTS2020_npy" \
    --dataname BRATS2020 \
    --split_config "$SPLIT_CONFIG" \
    --data_seed "$DATA_SEED" \
    --seed 42 \
    --lam_sd 0.1 \
    --compute_hd95 \
    --compute_pers_gain \
    --eval_global_model \
    --version "pmcs_split${SPLIT_ID}_smoke_${SLURM_JOB_ID}" \
    --resume 0
done

python scripts/audit_split_smoke.py \
  "results/pmcs_splitA_smoke_${SLURM_JOB_ID}" \
  "results/pmcs_splitB_smoke_${SLURM_JOB_ID}"

echo "PAIRED SPLIT SMOKE TEST COMPLETED"
