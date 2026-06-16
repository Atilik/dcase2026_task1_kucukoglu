#!/bin/bash
#SBATCH --job-name=ext_strategies
#SBATCH --account=torch_pr_92_general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/strategies_%j.out
#SBATCH --error=logs/strategies_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu

# Initialize singularity environment
export SINGULARITY_EXEC="./sing"

echo "======================================"
echo " Building Combined Dataset "
echo "======================================"
$SINGULARITY_EXEC <<< "python -u build_combined_dataset.py"

echo "======================================"
echo " Strategy A: Frozen Embeddings "
echo "======================================"
echo "--- Running ext_weak ---"
$SINGULARITY_EXEC <<< "python -u train_test.py --config config_ext_weak.yaml"

echo "--- Running ext_all ---"
$SINGULARITY_EXEC <<< "python -u train_test.py --config config_ext_all.yaml"

echo "--- Running ext_weighted ---"
$SINGULARITY_EXEC <<< "python -u train_test.py --config config_ext_weighted.yaml"

echo "======================================"
echo " Strategy B: Fine-tune CLAP "
echo "======================================"
# Note: FOLD_ONLY=0 is set in the script right now to test.
$SINGULARITY_EXEC <<< "python -u finetune_clap_external.py"

echo "All strategies completed."
