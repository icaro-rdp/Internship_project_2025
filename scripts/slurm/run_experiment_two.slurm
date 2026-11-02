#!/bin/bash
#SBATCH --job-name=exp2_cluster
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=all-compute-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# SLURM batch script for Experiment 2 (Explainability Map Generation & Comparison).
# Stages the dataset to local scratch storage, configures thread pinning and the
# project virtual environment, and executes the Experiment 2 pipeline on GPU.
# Custom CLI arguments can be passed via `sbatch run_experiment_two.slurm <args>`.

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
echo "HPC Cluster - Experiment 2 Execution"
echo "=========================================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $(hostname)"
echo "CPUs allocated:  $SLURM_CPUS_PER_TASK"
echo "GPU allocated:   ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L
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

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs Outputs/Experiment_2_variants/XAI_Maps/GradCAM \
    Outputs/Experiment_2_variants/XAI_Maps/Multiscale_Pixel_Masking \
    Outputs/Experiment_2_variants/XAI_Maps/LIME

# --- 5. Execute Experiment 2 ---
# If no arguments are provided, run all models, unpruned variants, all XAI methods,
# and full comparisons with cluster optimizations (existing maps are skipped automatically).
if [ $# -eq 0 ]; then
    echo "Running with default arguments: all models, unpruned variants, all XAI methods, and comparisons"
    srun python -m Image_Authenticity_prediction experiment-two \
        --variants unpruned \
        --xai-methods all \
        --cluster-computing \
        --save-maps \
        --run-comparison \
        --comparison-kinds within_model_variants between_model_architectures cross_methods \
        --comparison-metrics correlation ssim rmse cosine top_percent_iou_10 earth_movers_distance
else
    echo "Running with custom arguments: $@"
    srun python -m Image_Authenticity_prediction experiment-two "$@"
fi

echo "Experiment 2 execution completed successfully."
