#!/bin/bash
#SBATCH --job-name=eval_inf
#SBATCH --account=torch_pr_92_general
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/eval_inference_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/eval_inference_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs

echo "=== DCASE 2026 Task 1 Eval Inference ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start: $(date)"

./sing <<'EOF'
# Handle missing eval embeddings
python -c "
import numpy as np, os
sid = '8ebdf5bf-70ab-46b6-8fda-1420f3c485e1'
for d, dim in [('data/eval/features/clap_audio_embeddings', 512), ('data/eval/features/clap_text_embeddings', 512), ('data/eval/features/convnext_audio_embeddings', 768), ('data/eval/features/whisper_audio_embeddings', 512)]:
    p = os.path.join(d, f'{sid}.npy')
    if not os.path.exists(p):
        os.makedirs(d, exist_ok=True)
        np.save(p, np.zeros(dim, dtype='float32'))
        print(f'Created zero embedding: {p}')
"

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# Run full pipeline for sub 1 only (will extract clap_ft + inference)
python -u eval_inference.py --sub 1
EOF

echo "=== Finished: $(date) ==="
