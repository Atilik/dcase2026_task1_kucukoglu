#!/bin/bash
#SBATCH --job-name=contrastive
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/contrastive_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/contrastive_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs
./sing <<'EOF'
echo "=== 1/3: Contrastive λ=1.0 ==="
python -u main.py --config config_contrastive_10.yaml

echo "=== 2/3: Contrastive λ=0.5 ==="
python -u main.py --config config_contrastive_05.yaml

echo "=== 3/3: Contrastive λ=0.2 ==="
python -u main.py --config config_contrastive_02.yaml
EOF
