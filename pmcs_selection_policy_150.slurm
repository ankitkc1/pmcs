#!/bin/bash
#SBATCH --job-name=pmcs-selection
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa4000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=pmcs-selection-%j.out
#SBATCH --error=pmcs-selection-%j.err

set -euo pipefail

# 150-round partial-participation run under a chosen client-selection policy.
#
# Usage: sbatch pmcs_selection_policy_150.slurm <policy> <seed> <K> <scenario>
#   policy:   uniform | poc | mics
#   seed:     model seed (--seed; independent of --data_seed, the patient split)
#   K:        clients per round (--clients_per_round), out of --client_num 8
#   scenario: A | B  (Split A: T1ce scarce (n=2); Split B: FLAIR scarce (n=2))
#
# poc/mics use options.py's own defaults (poc_d=2*K, mics_beta=0.15,
# mics_smax=12 -- from the offline CPU sweep in scripts/mics_finalise.py, not
# tuned per run) unless overridden by exporting POC_D / MICS_BETA / MICS_SMAX
# before calling sbatch.
#
# Example: sbatch pmcs_selection_policy_150.slurm mics 42 2 A

POLICY="${1:?policy required: uniform|poc|mics}"
MODEL_SEED="${2:?seed required}"
K="${3:?K required}"
SCENARIO="${4:?scenario required: A|B}"

case "$POLICY" in
  uniform|poc|mics) ;;
  *) echo "unknown policy: $POLICY (expected uniform|poc|mics)" >&2; exit 1 ;;
esac
case "$SCENARIO" in
  A|B) ;;
  *) echo "unknown scenario: $SCENARIO (expected A|B)" >&2; exit 1 ;;
esac

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pmcs
cd "$HOME/pmcs"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

DATA_SEED=20260905
SPLIT_CONFIG="split/generated/data_seed_${DATA_SEED}/split${SCENARIO}/materialized_config.json"
VERSION="pmcs_sel_split${SCENARIO}_${POLICY}_K${K}_150r_seed${MODEL_SEED}_data${DATA_SEED}_job${SLURM_JOB_ID}"
RESULT_DIR="results/${VERSION}"

test -f "$SPLIT_CONFIG"

python -m py_compile \
  options.py train_federated.py dataset/datasets.py utils/metrics.py utils/selection.py \
  scripts/audit_full_run.py

# CPU-only, no data, no GPU -- must pass before a single GPU-hour is spent.
# This is the whole point of tests/test_selection.py: everything it checks
# (reproduction, simulator parity, mics touching no client-side quantity,
# well-formedness, the all-ones refutation) is verifiable ahead of time.
python tests/test_selection.py

SELECTION_OVERRIDES=""
if [ "$POLICY" = "poc" ] && [ -n "${POC_D:-}" ]; then
  SELECTION_OVERRIDES="--poc_d ${POC_D}"
elif [ "$POLICY" = "mics" ]; then
  SELECTION_OVERRIDES="--mics_beta ${MICS_BETA:-0.15} --mics_smax ${MICS_SMAX:-12}"
fi

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
  --clients_per_round "$K" \
  --selection_policy "$POLICY" \
  $SELECTION_OVERRIDES \
  --lam_sd 0.1 \
  --compute_hd95 \
  --compute_pers_gain \
  --eval_global_model \
  --version "$VERSION" \
  --resume 0

python scripts/audit_full_run.py "$RESULT_DIR" \
  --split "$SCENARIO" --rounds 150 --eval_every 10

echo "SELECTION POLICY RUN COMPLETED: policy=$POLICY seed=$MODEL_SEED K=$K scenario=$SCENARIO"
echo "selection_log.jsonl: ${RESULT_DIR}/selection_log.jsonl"
