#!/bin/bash
#SBATCH --job-name=pmcs-coslr50-v2
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa4000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=pmcs-coslr50-v2-%j.out
#SBATCH --error=pmcs-coslr50-v2-%j.err

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pmcs

cd "$HOME/pmcs"

# Reduce CUDA allocator fragmentation.
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Stop before training if the repaired trainer is invalid.
python -m py_compile train_federated.py
echo "PMCS syntax OK — starting training"

python -u train_federated.py \
  --client_num 8 \
  --gpus 0,1,2,3 \
  --batch_size 1 \
  --local_ep 1 \
  --c_rounds 50 \
  --eval 10 \
  --datapath "$HOME/FedMEPD/data/BraTS2020_npy" \
  --dataname BRATS2020 \
  --setting_options c8 \
  --lam_sd 0.1 \
  --version pmcs_brats20_50rounds_cosine_lr_v2 \
  --resume 0
