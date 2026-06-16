#!/bin/bash
#SBATCH --job-name=concat_exp3b
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/concat_exp3b_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/concat_exp3b_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu

set -euo pipefail

PROJ_DIR=/scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p "$PROJ_DIR/logs"

cd "$PROJ_DIR"

./sing <<'EOF'
echo ""
echo "=============================================="
echo "  DCASE 2026 Task 1 - Experiment 3b"
echo "  CLAP+BEATs audio (1280-dim) + CLAP text"
echo "  Config: config_concat.yaml"
echo "=============================================="
echo ""

# Force CPU - HATR model is too small to benefit from GPU
export CUDA_VISIBLE_DEVICES=""

# Step 1: Concatenate embeddings (fast, ~30 seconds)
echo "--- Concatenating CLAP + BEATs audio embeddings ---"
python -u concat_embeddings.py
echo ""

# Step 2: Build dataset + Train + Summarize
echo "--- Running training pipeline ---"
python -u main.py --config config_concat.yaml

echo ""
echo "Done! Check model_output_concat/ for results."
EOF
