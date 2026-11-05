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
# Pattern: stage input & existing maps to /scratch -> run on NVMe -> copy output home -> automatic cleanup.
# Custom CLI arguments can be passed via `sbatch scripts/slurm/run_experiment_two.slurm <args>`.

set -euo pipefail

# Validate identifiers before constructing a removable path
if [[ -z "${USER:-}" || "${USER:-}" == */* ||
      ! "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]]; then
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

echo "=========================================================="
echo "HPC Cluster - Experiment 2 Execution"
echo "=========================================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Node:            $(hostname)"
echo "CPUs allocated:  $SLURM_CPUS_PER_TASK"
echo "GPU allocated:   ${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi -L
echo "=========================================================="

# --- 1. Environment modules & thread pinning ---
source /opt/ohpc/admin/lmod/lmod/init/bash
source /etc/profile.d/zz-ohpc-modulepath.sh
module purge

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# --- 2. Activate Project .venv ---
source "$SLURM_SUBMIT_DIR/.venv/bin/activate"

# --- 3. Stage Dataset & Existing XAI Maps to local NVMe /scratch ---
DATA_SRC="${DATA_DIR:-$SLURM_SUBMIT_DIR/Image_Authenticity_prediction/Dataset/AIGCIQA2023}"
if [ ! -d "$DATA_SRC" ]; then
    echo "Input not found: $DATA_SRC (prepare it on the login node)" >&2
    exit 2
fi
mkdir -p "$SCRATCH/Dataset/AIGCIQA2023"
echo "Staging dataset to $SCRATCH/Dataset/AIGCIQA2023..."
cp -a "$DATA_SRC/." "$SCRATCH/Dataset/AIGCIQA2023/"

EXP2_SCRATCH="$SCRATCH/Outputs/Experiment_2_variants"
mkdir -p "$EXP2_SCRATCH/XAI_Maps/GradCAM" \
    "$EXP2_SCRATCH/XAI_Maps/Multiscale_Pixel_Masking" \
    "$EXP2_SCRATCH/XAI_Maps/LIME" \
    "$EXP2_SCRATCH/Plots"

echo "Staging existing XAI maps to $EXP2_SCRATCH/XAI_Maps..."
if [ -d "$SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants/XAI_Maps/GradCAM" ]; then
    cp -a "$SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants/XAI_Maps/GradCAM/"*_maps.npy \
        "$EXP2_SCRATCH/XAI_Maps/GradCAM/" 2>/dev/null || true
fi
if [ -d "$SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants/XAI_Maps/Multiscale_Pixel_Masking" ]; then
    cp -a "$SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants/XAI_Maps/Multiscale_Pixel_Masking/"*_maps.npy \
        "$EXP2_SCRATCH/XAI_Maps/Multiscale_Pixel_Masking/" 2>/dev/null || true
fi
if [ -d "$SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants/XAI_Maps/LIME" ]; then
    cp -a "$SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants/XAI_Maps/LIME/." \
        "$EXP2_SCRATCH/XAI_Maps/LIME/"
fi
if [ -f "$SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants/XAI_Maps/test_samples_manifest.json" ]; then
    cp -a "$SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants/XAI_Maps/test_samples_manifest.json" \
        "$EXP2_SCRATCH/XAI_Maps/"
fi

export DATASET_ROOT="$SCRATCH"
export EXP2_OUTPUT_DIR="$EXP2_SCRATCH"

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs \
    Outputs/Experiment_2_variants/XAI_Maps/GradCAM \
    Outputs/Experiment_2_variants/XAI_Maps/Multiscale_Pixel_Masking \
    Outputs/Experiment_2_variants/XAI_Maps/LIME \
    Outputs/Experiment_2_variants/Plots

# --- 4. Execute Experiment 2 on local NVMe /scratch ---
if [ $# -eq 0 ]; then
    echo "Step 1/2: Generating missing LIME maps on local /scratch..."
    srun python -m Image_Authenticity_prediction experiment-two \
        --variants unpruned \
        --xai-methods lime \
        --cluster-computing \
        --save-maps

    echo "Syncing generated LIME maps from /scratch to shared storage..."
    cp -a "$EXP2_SCRATCH/XAI_Maps/LIME/." \
        "$SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants/XAI_Maps/LIME/"

    echo "Step 2/2: Running full comparisons across all XAI methods on local /scratch..."
    srun python -m Image_Authenticity_prediction experiment-two \
        --variants unpruned \
        --xai-methods all \
        --comparison-only \
        --save-plots \
        --comparison-kinds within_model_variants between_model_architectures cross_methods \
        --comparison-metrics correlation ssim top_percent_iou_5 top_percent_iou_15 top_percent_iou_25
else
    echo "Running with custom arguments: $@"
    srun python -m Image_Authenticity_prediction experiment-two "$@"
fi

# --- 5. Copy results to shared storage only after success, before the trap cleans up ---
echo "Copying results from $EXP2_SCRATCH to $SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants..."
cp -a "$EXP2_SCRATCH/XAI_Maps/LIME/." \
    "$SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants/XAI_Maps/LIME/"
if [ -f "$EXP2_SCRATCH/XAI_Maps/test_samples_manifest.json" ]; then
    cp -a "$EXP2_SCRATCH/XAI_Maps/test_samples_manifest.json" \
        "$SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants/XAI_Maps/"
fi
if [ -d "$EXP2_SCRATCH/Plots" ]; then
    cp -a "$EXP2_SCRATCH/Plots/." \
        "$SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants/Plots/"
fi
if [ -f "$EXP2_SCRATCH/experiment_2b_comparison.json" ]; then
    cp -a "$EXP2_SCRATCH/experiment_2b_comparison.json" \
        "$SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants/"
fi

echo "Results copied to: $SLURM_SUBMIT_DIR/Outputs/Experiment_2_variants"
echo "Experiment 2 execution completed successfully."
