#!/bin/bash
#SBATCH --job-name=pose_lifter_ddp
#SBATCH --nodes=8               # 8 nodes × 4 GPUs = 32 GPUs
#SBATCH --ntasks-per-node=4     # one rank per GPU
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --output=LogsLifter/out/%j.out
#SBATCH --error=LogsLifter/err/%j.err
#SBATCH --hint=nomultithread
#SBATCH --exclusive            # recommended when you grab >1 node

module purge
module load pytorch-gpu/py3/2.6.0

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

srun python -u train_lifter_ddp.py \
      --data $SCRATCH/smpl/amass/smplh/sixteenth-training-data \
      --epochs 50 \
      --batch 64 \
      --window 13 \
      --amp \
      --checkpoint ./ckpt_epoch01_step006000.pt\
      --save-every 1000 

