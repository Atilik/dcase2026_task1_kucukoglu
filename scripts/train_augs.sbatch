#!/bin/bash
#SBATCH --job-name=augs
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/augs_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/augs_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs
./sing <<'EOF'
echo "=== 1/4: Label smoothing 0.1 ==="
python -u main.py --config config_ls01.yaml

echo "=== 2/4: CutMix α=0.2 ==="
python -u main.py --config config_cutmix02.yaml

echo "=== 3/4: Cross-modal swap 0.3 + noise 0.001 ==="
python -u main.py --config config_xswap_noise.yaml

echo "=== 4/4: Combo: mixup + ls + xswap + noise ==="
python -u main.py --config config_combo_aug.yaml
EOF
