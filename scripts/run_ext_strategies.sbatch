#!/bin/bash
#SBATCH --job-name=ext_all3
#SBATCH --account=torch_pr_92_general
#SBATCH --partition=cs
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/repos/dcase2026_task1_baseline/logs/ext_strategies_%j.out
#SBATCH --error=/scratch/%u/repos/dcase2026_task1_baseline/logs/ext_strategies_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mk9649@nyu.edu
set -euo pipefail
cd /scratch/$USER/repos/dcase2026_task1_baseline
mkdir -p logs

echo "================================================================"
echo "EXTERNAL DATA STRATEGIES — ALL 3 + ENSEMBLE"
echo "Start: $(date)"
echo "================================================================"

# ── Step 0: Check test set class distribution ──
./sing <<'EOF'
echo ""
echo "=== Step 0: Test Set Distribution ==="
python -u check_test_distribution.py
EOF

# ── Strategy A1: Weak classes only ──
./sing <<'EOF'
echo ""
echo "================================================================"
echo "=== Strategy A1: External Weak Classes Only ==="
echo "================================================================"
echo "Start: $(date)"
CONFIG_FILE=config_ext_weak.yaml python -u main.py --config config_ext_weak.yaml
echo "End: $(date)"
EOF

# ── Strategy A2: All external classes ──
./sing <<'EOF'
echo ""
echo "================================================================"
echo "=== Strategy A2: External ALL Classes ==="
echo "================================================================"
echo "Start: $(date)"
CONFIG_FILE=config_ext_all.yaml python -u main.py --config config_ext_all.yaml
echo "End: $(date)"
EOF

# ── Strategy A3: Weighted external ──
./sing <<'EOF'
echo ""
echo "================================================================"
echo "=== Strategy A3: External Weighted ==="
echo "================================================================"
echo "Start: $(date)"
CONFIG_FILE=config_ext_weighted.yaml python -u main.py --config config_ext_weighted.yaml
echo "End: $(date)"
EOF

# ── Ensemble evaluation (include new models) ──
./sing <<'EOF'
echo ""
echo "================================================================"
echo "=== Ensemble Evaluation (all models) ==="
echo "================================================================"
echo "Start: $(date)"
python -u ensemble_evaluate.py
echo "End: $(date)"
EOF

echo ""
echo "================================================================"
echo "ALL STRATEGIES COMPLETED: $(date)"
echo "================================================================"
