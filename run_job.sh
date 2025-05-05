#!/bin/bash
#SBATCH --job-name=pose_lifter       # a name you choose
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10           # matches your DataLoader num_workers
#SBATCH --gres=gpu:1                 # request 1 GPU
#SBATCH --hint=nomultithread         # disable hyperthreads
#SBATCH --time=20:00:00              # adjust as needed (max 20h)
#SBATCH --output=LogsLifter/out/%j.out
#SBATCH --error=LogsLifter/err/%j.err

# --- load modules ---
module purge
module load pytorch-gpu/py3/2.6.0    # gives you torch, cuda, cudnn, etc.

# (optional) if you have a virtualenv or conda env:
# source $WORK/env-lifter/bin/activate

# --- go to your code directory ---
cd $WORK/pose-lifter

# --- run training ---
set -x
python train_lifter.py \
    --data /lustre/fswork/projects/rech/bwd/ugc42dc/pose-lifter/smpl/amass/smplh/test_npz/ \
    --epochs 50 \
    --batch 32 \
    --checkpoint ./best_lifter03.pt   # if you want to resume
