#!/bin/bash
#SBATCH --job-name=wh35k
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/whisper_bsd35k_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/whisper_bsd35k_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs
./sing <<'EOF'
echo "=== Extracting Whisper embeddings for BSD35k-CS ==="
python -u extract_whisper_embeddings.py \
    --audio-dir data/BSD35k-CS/audio \
    --output-dir data/BSD35k-CS/features/whisper_audio_embeddings
EOF
