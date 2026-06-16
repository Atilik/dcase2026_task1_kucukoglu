#!/bin/bash
#SBATCH --job-name=wh_bl
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/whisper_train_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/whisper_train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs
./sing <<'EOF'
echo "=== Whisper baseline ==="
python -u main.py --config config_whisper.yaml
EOF
