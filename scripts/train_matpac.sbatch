#!/bin/bash
#SBATCH --job-name=matpac
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/matpac_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/matpac_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs

MATPAC_ENV="/scratch/$USER/envs/matpac_env"
MATPAC_CKPT="/scratch/$USER/shared/dcase2026_task5/dataset_copy/model_checkpoints/matpac/matpac_plus_6s_2048_enconly.pt"

# ── Step 1: Extract embeddings using MATPAC conda env (NO Singularity needed) ──
echo "=== Extracting MATPAC embeddings ==="
eval "$($MATPAC_ENV/bin/conda shell.bash hook 2>/dev/null || true)"
export PATH="$MATPAC_ENV/bin:$PATH"
python -u extract_matpac_embeddings.py --checkpoint "$MATPAC_CKPT"
echo "=== Extraction done ==="

# ── Step 2: Build dataset + train (main env via Singularity read-only) ──
./sing <<'EOF'
echo "=== Building MATPAC dataset ==="
CONFIG_FILE=config_matpac.yaml python -u build_dataset.py

echo "=== Training MATPAC model ==="
CONFIG_FILE=config_matpac.yaml python -u main.py --config config_matpac.yaml
EOF
