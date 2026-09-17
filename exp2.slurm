#!/bin/bash
# =====================================================================
# Experiment 2 — one selector per array task, one seed.
#
#   sbatch --array=0-3 exp2.slurm          # the four priority runs
#   sbatch --array=0   exp2.slurm          # cpow-d only, if budget is tight
#
# Array index -> policy, so each run gets its own node and its own log.
# =====================================================================
#SBATCH --job-name=exp2
#SBATCH --output=logs/exp2_%A_%a.out
#SBATCH --error=logs/exp2_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=gpu          # <-- check your cluster's partition name

set -euo pipefail
mkdir -p logs exp2

module purge
module load cuda                  # <-- match what your existing SLURM uses
source ~/pmcs/venv/bin/activate   # <-- your environment

cd ~/pmcs

# ---------------------------------------------------------------- config
SCEN=${SCEN:-A}
K=${K:-2}
SEED=${SEED:-42}
ROUNDS=${ROUNDS:-150}

# index -> (policy, extra args, rounds)
# MFedMC trains ALL clients every round, so it costs ~4x a K=2 run.
# Its coverage result is structural and visible early -- 30 rounds is enough
# to show the upload filter starving an encoder. Raise it only if you need
# Dice from that arm.
case ${SLURM_ARRAY_TASK_ID} in
  0) POLICY=powd    ; EXTRA="--d_mult 2 --cpow"        ; R=${ROUNDS} ;;
  1) POLICY=rpowd   ; EXTRA="--d_mult 2"               ; R=${ROUNDS} ;;
  2) POLICY=mfedmc  ; EXTRA="--gamma 1 --delta 0.2"    ; R=30        ;;
  3) POLICY=mfedmc  ; EXTRA="--gamma 4 --delta 0.2"    ; R=30        ;;
  *) echo "bad array index"; exit 1 ;;
esac

echo "=========================================================="
echo " policy   ${POLICY} ${EXTRA}"
echo " scenario ${SCEN}   K=${K}  seed=${SEED}  rounds=${R}"
echo " node     $(hostname)   gpu $(nvidia-smi --query-gpu=name \
        --format=csv,noheader | head -1)"
echo " started  $(date)"
echo "=========================================================="

srun python -u train_federated.py \
    --policy      "${POLICY}" \
    --scenario    "${SCEN}" \
    --clients     8 \
    --K           "${K}" \
    --rounds      "${R}" \
    --seed        "${SEED}" \
    --selection-seed-offset 10000 \
    --log-selection exp2/ \
    ${EXTRA}

echo " finished $(date)"
