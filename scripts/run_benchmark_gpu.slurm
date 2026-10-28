#!/bin/bash
#SBATCH --job-name=bench_mpm_blackwell
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=all-compute-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

# DIPSEN HPC Golden Rules compliance:
# 1. Explicit request: partition, gres, cpus-per-task, mem, time
# 2. Even CPU count: 8 (SMT core allocation)
# 3. Validated /scratch staging and cleanup trap from dipsen-hpc skill
# 4. Thread pinning matching allocated CPUs
# 5. Environment activation from pre-built .venv (no WAN access in jobs)

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
    rm -rf -- "$SCRATCH"
}
trap cleanup_scratch EXIT

echo "=== Job Info ==="
echo "Node:            $(hostname)"
echo "Job ID:          $SLURM_JOB_ID"
echo "CPUs allocated:  $SLURM_CPUS_PER_TASK"
echo "GPU allocated:   ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L

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

# --- 4. Stage Dataset Sample to local NVMe /scratch ---
mkdir -p "$SCRATCH/Dataset"
cp -a "$SLURM_SUBMIT_DIR/Image_Authenticity_prediction/Dataset/AIGCIQA2023" "$SCRATCH/Dataset/"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# --- 5. Execute Optimization Benchmark ---
python scripts/benchmark_mpm_optimizations.py \
    --weights "Outputs/Experiment_1_variants/Weights/densenet161_exp1a_variant1_best.pth" \
    --dataset-dir "$SCRATCH/Dataset/AIGCIQA2023" \
    --sigmas 4 8 16

echo "Benchmark completed successfully."
