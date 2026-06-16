#!/bin/bash
#SBATCH --job-name=ft_emb
#SBATCH --account=torch_pr_92_general
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/ft_emb_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/ft_emb_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs
./sing <<'EOF'
echo "=== Extracting Fine-tuned CLAP Embeddings ==="
echo "Clearing stale embeddings from previous runs..."
rm -rf data/BSD10k-v1.2/features/finetuned_clap_audio_embeddings
echo "Start: $(date)"
python -u extract_finetuned_embeddings.py
echo "End: $(date)"

echo ""
echo "=== Symlinking fold_0 embeddings into base dir for build_dataset.py ==="
EMB_DIR=data/BSD10k-v1.2/features/finetuned_clap_audio_embeddings
for f in "$EMB_DIR"/fold_0/*.npy; do
    bn=$(basename "$f")
    ln -sf "fold_0/$bn" "$EMB_DIR/$bn"
done
echo "Created $(ls "$EMB_DIR"/*.npy 2>/dev/null | wc -l) symlinks"

echo ""
echo "=== Clearing stale multimodal output ==="
rm -rf model_output_finetune_mm data_finetune

echo ""
echo "=== Training Multimodal HATR on Fine-tuned Embeddings ==="
echo "Start: $(date)"
CONFIG_FILE=config_finetune.yaml python -u main.py --config config_finetune.yaml
echo "End: $(date)"
EOF
