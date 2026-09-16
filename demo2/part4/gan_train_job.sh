#!/bin/bash

#SBATCH --job-name=gan-oasis
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=demo2/part4/gan_train_%j.out
#SBATCH --error=demo2/part4/gan_train_%j.err

echo "============================================================"
echo "COMP3710 DEMO 2 - PART 4 TASK 3"
echo "OASIS GAN TRAINING"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "============================================================"

source $HOME/miniconda3/bin/activate
conda activate torch

cd $HOME/comp3710

nvidia-smi

python -u demo2/part4/gan_oasis.py \
    --epochs 40 \
    --batch-size 32 \
    --latent-dim 128 \
    --lr 0.0002 \
    --num-workers 4

echo "============================================================"
echo "Finished: $(date)"
echo "============================================================"
