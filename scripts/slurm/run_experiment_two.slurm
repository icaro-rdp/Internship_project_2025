#!/bin/bash
#SBATCH --job-name=exp2_cluster
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=all-compute-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# DIPSEN HPC Golden Rules compliance:
# 1. Explicit request: partition, gres, cpus-per-task, mem, time
# 2. Even CPU count: 8 (SMT core allocation)
# 3. Validated /scratch staging and cleanup trap from dipsen-hpc skill
# 4. Thread pinning matching allocated CPUs
# 5. Environment activation from pre-built .venv (no WAN access in jobs)
# 6. Forwards all CLI arguments to experiment-two via "$@"

set -euo pipefail

# --- 1. Validated /scratch staging on c31 (Rule 8) ---
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
echo "DIPSEN HPC (Node c31) - Experiment 2 Execution"
echo "=========================================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $(hostname)"
echo "CPUs allocated:  $SLURM_CPUS_PER_TASK"
echo "GPU allocated:   ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L
echo "=========================================================="

# --- 2. Lmod Environment & Thread Pinning (Rules 4 & 6) ---
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
mkdir -p logs Outputs/Experiment_2_variants/XAI_Maps

# --- 5. Forward CLI arguments to experiment-two ---
# If no arguments provided, default to running all remaining variants with cluster optimizations
if [ $# -eq 0 ]; then
    echo "Running with default arguments: remaining models (densenet161, efficientnetb3) with --cluster-computing"
    srun python -m Image_Authenticity_prediction experiment-two \
        --models densenet161 efficientnetb3 \
        --variants unpruned \
        --xai-methods both \
        --cluster-computing \
        --save-maps
else
    echo "Running with custom arguments: $@"
    srun python -m Image_Authenticity_prediction experiment-two "$@"
fi

echo "Experiment 2 execution completed successfully."
