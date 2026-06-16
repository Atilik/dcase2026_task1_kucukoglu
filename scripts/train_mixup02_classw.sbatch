#!/bin/bash
#SBATCH --job-name=m02_clw
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/mixup02_classw_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/mixup02_classw_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs
./sing <<'EOF'
echo "=== Mixup 0.2 + class weights ==="
python -u main.py --config config_mixup02_classw.yaml
EOF
