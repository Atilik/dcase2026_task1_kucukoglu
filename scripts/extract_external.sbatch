#!/bin/bash
#SBATCH --job-name=ext_clap
#SBATCH --account=torch_pr_92_general
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu

echo "=== Phase 2: Extract External CLAP Embeddings ==="
echo "Start: $(date)"

./sing <<< "python -u extract_external_clap.py"

echo ""
echo "=== Done: $(date) ==="
