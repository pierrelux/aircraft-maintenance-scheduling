#!/bin/bash
#SBATCH --job-name=maintainer-train
#SBATCH --partition=long
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# ============================================================================
# Maintainer v1 Training Pipeline for MILA Cluster
# ============================================================================
# This script:
#   1. Sets up a conda environment with PyTorch 2.0+ and dependencies
#   2. Generates training data from Deng's Excel dataset
#   3. Trains MaintainerNet via behavior cloning
#   4. Evaluates the trained model
# ============================================================================

set -e  # Exit on error

# Configuration
PROJECT_DIR="/home/mila/p/pierre-luc.bacon/aircraft-maintenance-scheduling-1"
ENV_NAME="maintainer"
DATA_DIR="${PROJECT_DIR}/data"
CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints"
LOG_DIR="${PROJECT_DIR}/logs"

# Deng Excel dataset
EXCEL_FILE="${DATA_DIR}/Sep 25 2017 (Case Study 2018-2021).xlsx"

# Training hyperparameters
N_TRAIN_INSTANCES=500  # Increased for better generalization
N_EVAL_INSTANCES=50
EPOCHS=50
BATCH_SIZE=32
D_MODEL=256
N_LAYERS=6
LR=1e-4

echo "=============================================="
echo "MAINTAINER v1 TRAINING PIPELINE"
echo "=============================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURMD_NODENAME}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Start time: $(date)"
echo "=============================================="

# Create directories
mkdir -p "${DATA_DIR}" "${CHECKPOINT_DIR}" "${LOG_DIR}"

# Load miniconda module
echo ""
echo "[1/6] Loading modules..."
module load miniconda/3

# Create or activate conda environment
echo ""
echo "[2/6] Setting up conda environment..."
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  Activating existing environment: ${ENV_NAME}"
    source activate ${ENV_NAME}
    
    # Install wandb and python-dotenv if not already installed
    pip install wandb python-dotenv --quiet
else
    echo "  Creating new environment: ${ENV_NAME}"
    conda create -n ${ENV_NAME} python=3.10 -y
    source activate ${ENV_NAME}
    
    echo "  Installing PyTorch with CUDA..."
    pip install --upgrade pip
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    echo "  Installing project dependencies..."
    pip install -r "${PROJECT_DIR}/requirements.txt"
fi

# Load environment variables from .env
if [ -f "${PROJECT_DIR}/.env" ]; then
    echo "  Loading environment variables from .env..."
    export $(cat "${PROJECT_DIR}/.env" | xargs)
fi

# Verify GPU availability
echo ""
echo "[3/6] Verifying GPU..."
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Navigate to project directory
cd "${PROJECT_DIR}"

# Generate training data from Deng Excel dataset
echo ""
echo "[4/6] Generating training data..."
TRAIN_DATA="${DATA_DIR}/train_deng_${N_TRAIN_INSTANCES}.pkl"
if [ -f "${TRAIN_DATA}" ]; then
    echo "  Training data already exists: ${TRAIN_DATA}"
else
    if [ ! -f "${EXCEL_FILE}" ]; then
        echo "  ERROR: Deng Excel file not found: ${EXCEL_FILE}"
        echo "  Please download from: https://data.4tu.nl/ndownloader/items/5ec21947-7f4e-453b-a3c1-b7863b9d5cb7/versions/1"
        exit 1
    fi
    echo "  Generating ${N_TRAIN_INSTANCES} instances from Deng Excel data..."
    python -m maintainer.scripts.generate_data \
        --output "${TRAIN_DATA}" \
        --excel "${EXCEL_FILE}" \
        --n-instances ${N_TRAIN_INSTANCES}
fi

# Train the model
echo ""
echo "[5/6] Training MaintainerNet..."
echo "  Epochs: ${EPOCHS}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Model dim: ${D_MODEL}"
echo "  Layers: ${N_LAYERS}"
echo "  Learning rate: ${LR}"
# Check if we should resume from a checkpoint
RESUME_ARG=""
if [ -f "${CHECKPOINT_DIR}/checkpoint_latest.pt" ]; then
    echo "  Found existing checkpoint, will resume training..."
    RESUME_ARG="--resume latest"
fi

python -m maintainer.scripts.train \
    --data "${TRAIN_DATA}" \
    --output "${CHECKPOINT_DIR}/model.pt" \
    --epochs ${EPOCHS} \
    --batch-size ${BATCH_SIZE} \
    --d-model ${D_MODEL} \
    --n-layers ${N_LAYERS} \
    --lr ${LR} \
    --device cuda \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --checkpoint-interval 1 \
    ${RESUME_ARG}

# Evaluate the model
echo ""
echo "[6/6] Evaluating model..."
python -m maintainer.scripts.evaluate \
    --checkpoint "${CHECKPOINT_DIR}/model.pt" \
    --n-instances ${N_EVAL_INSTANCES}

echo ""
echo "=============================================="
echo "TRAINING COMPLETE"
echo "End time: $(date)"
echo "Checkpoint saved to: ${CHECKPOINT_DIR}/model.pt"
echo "=============================================="

