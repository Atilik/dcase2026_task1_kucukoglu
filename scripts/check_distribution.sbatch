#!/bin/bash
#SBATCH --job-name=chk_dist
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/check_dist_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/check_dist_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs
./sing <<'EOF'
echo "=== Test Set Class Distribution ==="
python -u check_test_distribution.py
EOF
