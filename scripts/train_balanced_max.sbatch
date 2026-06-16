#!/bin/bash
#SBATCH --job-name=bal_max
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/balanced_max_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/balanced_max_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs
./sing <<'EOF'
echo "=== Balanced BSD35k: equalize to max ==="
python -u main.py --config config_balanced_max.yaml
EOF
