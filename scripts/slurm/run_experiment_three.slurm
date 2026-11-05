#!/bin/bash
#SBATCH --job-name=exp3_ensemble
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=all-compute-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# SLURM batch script for Experiment 3 (Variant-Aligned Bagging & Stacking Ensembles + XAI).
# Evaluates 10 variant-aligned ensembles across 7 architectures, fits leakage-free
# validation meta-learners, and generates per-variant and prototype MPM heatmaps.
# Custom CLI arguments can be passed via `sbatch run_experiment_three.slurm <args>`.

set -euo pipefail

# --- 1. Local scratch staging and cleanup trap ---
if [[ -z "${USER:-}" || "${USER:-}" == */* || ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
    echo "Invalid user identity or SLURM_JOB_ID; scratch cleanup disabled." >&2
    exit 2
fi
readonly SCRATCH="/scratch/$USER/$SLURM_JOB_ID"
mkdir -p "$SCRATCH"

cleanup_scratch() {
    local expected="/scratch/$USER/$SLURM_JOB_ID"
    if [[ "${SCRATCH:-}" != "$expected" ]]; then
        echo "Cleanup refused for unexpected path: ${SCRATCH:-<not set>}" >&2
        return 0
    fi
    echo "[Cleanup] Removing local NVMe scratch: $expected"
    rm -rf -- "$SCRATCH"
}
trap cleanup_scratch EXIT

echo "=========================================================="
echo "HPC Cluster - Experiment 3 Execution"
echo "=========================================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $(hostname)"
echo "CPUs allocated:  $SLURM_CPUS_PER_TASK"
echo "GPU allocated:   ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L || true
echo "=========================================================="

# --- 2. Environment modules & thread pinning ---
source /opt/ohpc/admin/lmod/lmod/init/bash
source /etc/profile.d/zz-ohpc-modulepath.sh
module purge

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# --- 3. Activate Project .venv ---
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# --- 4. Stage Dataset to local NVMe /scratch ---
mkdir -p "$SCRATCH/Dataset"
echo "Staging dataset to /scratch..."
cp -a "$SLURM_SUBMIT_DIR/Image_Authenticity_prediction/Dataset/AIGCIQA2023" "$SCRATCH/Dataset/"
export DATASET_ROOT="$SCRATCH"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs \
    Outputs/Experiment_3_ensemble/Weights \
    Outputs/Experiment_3_ensemble/Results \
    Outputs/Experiment_3_ensemble/XAI_Maps/Multiscale_Pixel_Masking \
    Outputs/Experiment_3_ensemble/Plots

# --- 5. Execute Experiment 3 ---
# Default behavior runs both Bagging & Stacking across all 10 variants using precomputed
# Experiment 2 MPM maps (completes in <5 minutes).
# Pass `--live-mpm` to perform live forward passes across the test set.
if [ $# -eq 0 ]; then
    echo "Running with default arguments: all strategies, 10 variants, and Experiment 2 map reuse"
    srun python -m Image_Authenticity_prediction experiment-three \
        --strategy all \
        --variants all \
        --reuse-exp2-maps
else
    echo "Running with custom arguments: $@"
    srun python -m Image_Authenticity_prediction experiment-three "$@"
fi

echo "Experiment 3 execution completed successfully."
