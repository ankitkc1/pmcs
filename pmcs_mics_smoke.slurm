#!/bin/bash
#SBATCH --job-name=pmcs-mics-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:rtxa4000:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=pmcs-mics-smoke-%j.out
#SBATCH --error=pmcs-mics-smoke-%j.err
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pmcs
cd "$HOME/pmcs"
EXPECTED_SOURCE_COMMIT="676f5bbe61ee94f66c3174ed0eaaea28e0add784"
git cat-file -e "${EXPECTED_SOURCE_COMMIT}^{commit}" || {
    echo "ERROR: reviewed source commit is unavailable: $EXPECTED_SOURCE_COMMIT"
    exit 1
}
if ! git diff --quiet "$EXPECTED_SOURCE_COMMIT" -- \
    options.py train_federated.py dataset/datasets.py \
    utils/selection.py utils/metrics.py utils/encoder_coverage.py \
    scripts/audit_full_run.py tests/test_selection.py; then
    echo "ERROR: experiment source differs from reviewed commit $EXPECTED_SOURCE_COMMIT"
    exit 1
fi
DATA_SEED=20260905
SPLIT=A
K=2
MODEL_SEED=42
SPLIT_CONFIG="split/generated/data_seed_${DATA_SEED}/split${SPLIT}/materialized_config.json"
VERSION="pmcs_mics_smoke_split${SPLIT}_K${K}_seed${MODEL_SEED}_job${SLURM_JOB_ID}"
RESULT_DIR="results/${VERSION}"
test -f "$SPLIT_CONFIG" || { echo "ERROR: missing $SPLIT_CONFIG"; exit 1; }
test ! -e "$RESULT_DIR" || { echo "ERROR: result exists: $RESULT_DIR"; exit 1; }
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python -m py_compile options.py train_federated.py utils/selection.py scripts/audit_full_run.py
python tests/test_selection.py
python -u train_federated.py \
    --client_num 8 \
    --gpus 0,1 \
    --batch_size 1 \
    --local_ep 1 \
    --c_rounds 2 \
    --eval 2 \
    --datapath "$HOME/FedMEPD/data/BraTS2020_npy" \
    --dataname BRATS2020 \
    --split_config "$SPLIT_CONFIG" \
    --data_seed "$DATA_SEED" \
    --seed "$MODEL_SEED" \
    --clients_per_round "$K" \
    --selection_policy mics \
    --mics_beta 0.15 \
    --mics_smax 12 \
    --lam_sd 0.1 \
    --compute_hd95 \
    --compute_pers_gain \
    --eval_global_model \
    --version "$VERSION" \
    --resume 0
python scripts/audit_full_run.py "$RESULT_DIR" \
    --split "$SPLIT" --rounds 2 --eval_every 2
python - "$RESULT_DIR" <<'PY'
import json
import sys
from pathlib import Path
result = Path(sys.argv[1])
records = [json.loads(line) for line in (result / "selection_log.jsonl").read_text().splitlines() if line]
assert len(records) == 2
assert [row["round"] for row in records] == [0, 1]
assert all(row["policy"] == "mics" and row["K"] == 2 for row in records)
with (result / "metrics.json").open() as handle:
    metrics = json.load(handle)
assert metrics["static"]["config"]["selection_policy"] == "mics"
assert metrics["static"]["participation"]["K"] == 2
print("MICS GPU SMOKE PASSED")
print("Result directory:", result)
PY
