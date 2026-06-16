#!/bin/bash
#SBATCH --job-name=extract_logits
#SBATCH --account=torch_pr_92_general
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu

echo "Extracting logits for CLAP fine-tuning model"

# Run singularity container
./sing <<< "python -u extract_finetune_logits.py"
