#!/bin/bash
#SBATCH --job-name=aircraft-pcl
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=long

# Create logs directory if it doesn't exist
mkdir -p logs

# Load modules (adjust for your cluster)
module load python/3.10

# Activate virtual environment
source venv/bin/activate

# Set environment variables
export PYTHONUNBUFFERED=1

# Parse arguments
METHOD=${1:-transformer}  # transformer, bc, bc-finetune

echo "================================================"
echo "Starting training: $METHOD"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "================================================"

case $METHOD in
    transformer)
        python pcl_transformer.py \
            --iterations 1000 \
            --eval-every 100 \
            --d-model 256 \
            --n-heads 8 \
            --n-layers 4 \
            --lr 1e-4 \
            --entropy 0.2 \
            --batch-size 16 \
            --gradient-steps 16
        ;;
    bc)
        python bc_transformer.py \
            --bc-epochs 100 \
            --bc-lr 1e-3 \
            --n-expert-traj 30
        ;;
    bc-finetune)
        python bc_transformer.py \
            --bc-epochs 50 \
            --n-expert-traj 20 \
            --fine-tune \
            --fine-tune-iters 500
        ;;
    *)
        echo "Unknown method: $METHOD"
        echo "Usage: sbatch run_cluster.sh [transformer|bc|bc-finetune]"
        exit 1
        ;;
esac

echo "================================================"
echo "Training complete!"
echo "================================================"

