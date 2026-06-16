#!/bin/bash
#SBATCH --job-name=ext_map
#SBATCH --account=torch_pr_92_general
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu

echo "=== Phase 1: External Dataset Mapping ==="
echo "Start: $(date)"

./sing <<< "python -u build_external_dataset.py"

echo ""
echo "=== Done: $(date) ==="
