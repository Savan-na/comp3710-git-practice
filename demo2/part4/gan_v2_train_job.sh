#!/bin/bash

#SBATCH --job-name=gan-v2
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=demo2/part4/gan_v2_train_%j.out
#SBATCH --error=demo2/part4/gan_v2_train_%j.err

set -e

echo "============================================================"
echo "COMP3710 DEMO 2 - PART 4 TASK 3"
echo "OASIS GAN V2 - 64x64 WGAN-GP"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "============================================================"

source "$HOME/miniconda3/bin/activate"
conda activate torch
cd "$HOME/comp3710"

nvidia-smi

python - <<'PY'
import sys
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    sys.exit(2)
print("GPU:", torch.cuda.get_device_name(0))
PY

python -u demo2/part4/gan_oasis_v2.py \
    --epochs 60 \
    --batch-size 64 \
    --latent-dim 128 \
    --lr-g 0.0001 \
    --lr-c 0.0001 \
    --lambda-gp 10 \
    --n-critic 3 \
    --num-workers 4 \
    --save-every 10

echo "============================================================"
echo "GAN V2 JOB FINISHED"
echo "Finished: $(date)"
echo "============================================================"
