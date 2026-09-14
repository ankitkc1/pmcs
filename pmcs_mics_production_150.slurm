#!/bin/bash
#SBATCH --job-name=pmcs-mics-150
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa4000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-11%1
#SBATCH --output=pmcs-mics-150-%A_%a.out
#SBATCH --error=pmcs-mics-150-%A_%a.err
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pmcs
cd "$HOME/pmcs"
EXPECTED_SOURCE_COMMIT="676f5bbe61ee94f66c3174ed0eaaea28e0add784"
ACTUAL_COMMIT="$(git rev-parse HEAD)"
git cat-file -e "${EXPECTED_SOURCE_COMMIT}^{commit}" || {
    echo "ERROR: reviewed source commit is unavailable: $EXPECTED_SOURCE_COMMIT"
    exit 1
}
# HEAD may additionally contain this runner, but the reviewed experiment
# sources (including staged/unstaged content) must match the audited commit.
if ! git diff --quiet "$EXPECTED_SOURCE_COMMIT" -- \
    options.py train_federated.py dataset/datasets.py \
    utils/selection.py utils/metrics.py utils/encoder_coverage.py \
    scripts/audit_full_run.py tests/test_selection.py; then
    echo "ERROR: experiment source differs from reviewed commit $EXPECTED_SOURCE_COMMIT"
    git diff --stat "$EXPECTED_SOURCE_COMMIT" -- \
        options.py train_federated.py dataset/datasets.py \
        utils/selection.py utils/metrics.py utils/encoder_coverage.py \
        scripts/audit_full_run.py tests/test_selection.py
    exit 1
fi
SPLITS=(A A A A A A B B B B B B)
KS=(4 4 4 2 2 2 4 4 4 2 2 2)
SEEDS=(42 123 2026 42 123 2026 42 123 2026 42 123 2026)
TASK_ID="${SLURM_ARRAY_TASK_ID}"
SPLIT="${SPLITS[$TASK_ID]}"
K="${KS[$TASK_ID]}"
MODEL_SEED="${SEEDS[$TASK_ID]}"
DATA_SEED=20260905
BETA=0.15
S_MAX=12
SPLIT_CONFIG="split/generated/data_seed_${DATA_SEED}/split${SPLIT}/materialized_config.json"
VERSION="pmcs_mics_split${SPLIT}_K${K}_150r_seed${MODEL_SEED}_data${DATA_SEED}_job${SLURM_ARRAY_JOB_ID}_${TASK_ID}"
RESULT_DIR="results/${VERSION}"
test -f "$SPLIT_CONFIG" || {
    echo "ERROR: missing split configuration: $SPLIT_CONFIG"
    exit 1
}
test ! -e "$RESULT_DIR" || {
    echo "ERROR: refusing to overwrite existing result: $RESULT_DIR"
    exit 1
}
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
echo "Task=$TASK_ID Split=$SPLIT K=$K Seed=$MODEL_SEED"
echo "HEAD=$ACTUAL_COMMIT"
echo "Reviewed source=$EXPECTED_SOURCE_COMMIT"
echo "Result=$RESULT_DIR"
python -m py_compile \
    options.py train_federated.py dataset/datasets.py \
    utils/metrics.py utils/selection.py scripts/audit_full_run.py
python tests/test_selection.py
python -u train_federated.py \
    --client_num 8 \
    --gpus 0,1 \
    --batch_size 1 \
    --local_ep 1 \
    --c_rounds 150 \
    --eval 10 \
    --datapath "$HOME/FedMEPD/data/BraTS2020_npy" \
    --dataname BRATS2020 \
    --split_config "$SPLIT_CONFIG" \
    --data_seed "$DATA_SEED" \
    --seed "$MODEL_SEED" \
    --clients_per_round "$K" \
    --selection_policy mics \
    --mics_beta "$BETA" \
    --mics_smax "$S_MAX" \
    --lam_sd 0.1 \
    --compute_hd95 \