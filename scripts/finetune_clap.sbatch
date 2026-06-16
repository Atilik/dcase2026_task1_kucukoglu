#!/bin/bash
#SBATCH --job-name=clap_ft
#SBATCH --account=torch_pr_92_general
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/clap_ft_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/clap_ft_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs

./sing <<'EOF'
# First verify CLAP is installed and GPU works
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
import laion_clap
print('CLAP import OK')
"

echo "=== Fine-tuning CLAP ==="
python -u finetune_clap.py
EOF
