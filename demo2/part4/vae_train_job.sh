#!/bin/bash
#SBATCH --job-name=vae-oasis
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=demo2/part4/vae_train_%j.out
#SBATCH --error=demo2/part4/vae_train_%j.err

source $HOME/miniconda3/bin/activate
conda activate torch

cd $HOME/comp3710

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

nvidia-smi

python demo2/part4/vae_oasis.py

echo "Finished: $(date)"
