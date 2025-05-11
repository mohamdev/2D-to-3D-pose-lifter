#!/bin/bash
#SBATCH --job-name=pose_lifter_ddp
#SBATCH --nodes=1               # 8 nodes × 4 GPUs = 32 GPUs
#SBATCH --ntasks-per-node=1     # one rank per GPU
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:00:00
#SBATCH --output=LogsLifter/out/seg%j.out
#SBATCH --error=LogsLifter/err/seg%j.err
#SBATCH --hint=nomultithread
#SBATCH --exclusive            # recommended when you grab >1 node

module purge
module load pytorch-gpu/py3/2.1.1

export MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export MASTER_PORT=29500

srun python -u train_finetune_seg.py \
      --data $SCRATCH/smpl/amass/smplh/amass_3D_poses/train/ \
      --epochs 20 \
      --batch 64 \
      --window 13 \
      --amp \
      --checkpoint ./trained_models/best_lifter_128_ddp_epoch50.pt\
