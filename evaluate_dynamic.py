#!/usr/bin/env python3
"""
evaluate_dynamic.py

Load a .npz containing only 'joints_3d' (F,J,3), 
sample a random camera (intrinsics + extrinsics) exactly as in training,
project the full GT 3D → 2D in sliding windows,
run your TransformerLifter to lift 2D → 3D,
compute MPJPE against the GT,
and save a new .npz (--out) with arrays:
  - gt_joints:   (F,J,3) ground truth
  - pred_joints: (F,J,3) estimated
"""

import sys
import argparse
import random
import numpy as np
import torch

from train_lifter_ddp_dynamic_cams_tiny import (
    TransformerLifter,
    sample_direction, look_at_R,
    IMG_W, IMG_H, F_MIN, F_MAX, D_MAX, MARGIN, SEG_NORM
)

# joints & edges must match your data
JOINT_NAMES = [
    'r_shoulder','l_shoulder',
    'r_elbow',   'l_elbow',
    'r_wrist',   'l_wrist',
    'r_hip',     'l_hip',
    'r_knee',    'l_knee',
    'r_ankle',   'l_ankle',
]
SEG_PAIRS = [(2,4),(3,5),(0,1),(0,6),(6,8),(7,9)]  # same as DynamicPoseDataset

def compute_mpjpe(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(gt - pred, axis=-1)))

def predict_traj(
    j3d: np.ndarray,
    model, device,
    window: int,
    K: np.ndarray, R_w2c: np.ndarray, t_w2c: np.ndarray,
    seg_vec: np.ndarray
) -> np.ndarray:
    F, J, _ = j3d.shape
    pred_acc  = np.zeros((F, J, 3), dtype=np.float32)
    count_acc = np.zeros((F,1,1), dtype=np.float32)

    # normalized intrinsics
    f_norm  = K[0,0] / 2000.0
    cx_norm = (K[0,2] - IMG_W/2)/IMG_W
    cy_norm = (K[1,2] - IMG_H/2)/IMG_H
    k_vec = torch.tensor([f_norm, cx_norm, cy_norm],
                         dtype=torch.float32, device=device).unsqueeze(0)
    seg_t = torch.from_numpy(seg_vec.astype(np.float32)).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        for start in range(0, F - window + 1):
            end = start + window
            win3d = j3d[start:end]              # (T,J,3)
            X = win3d.reshape(-1,3).T           # (3, T*J)
            Xc = R_w2c @ X + t_w2c[:,None]      # (3, T*J)
            uvw = K @ Xc                        # (3, T*J)
            uv  = (uvw[:2]/(uvw[2:]+1e-8)).T     # (T*J,2)
            uv  = uv.reshape(window, J, 2) / IMG_W

            x2d = torch.from_numpy(uv.astype(np.float32))\
                       .unsqueeze(0).to(device)   # (1,T,J,2)
            pred3d = model(x2d, k_vec, seg_t)   # (1,T,J,3)
            pred  = pred3d[0].cpu().numpy()     # (T,J,3)

            pred_acc[start:end] += pred
            count_acc[start:end] += 1.0

    return (pred_acc / count_acc).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',    required=True,
                   help="Input .npz with 'joints_3d'")
    p.add_argument('--model',  required=True,
                   help="TransformerLifter checkpoint")
    p.add_argument('--window', type=int, default=13,
                   help="Sliding window size")
    p.add_argument('--seed',   type=int, default=0,
                   help="Random seed")
    p.add_argument('--out',    required=True,
                   help="Output .npz path")
    args = p.parse_args()

    # reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(args.npz)
    if 'joints_3d' not in data:
        print("Error: 'joints_3d' not found in", args.npz)
        sys.exit(1)
    j3d = data['joints_3d']  # (F,J,3)
    F,J,_ = j3d.shape

    # sample a random camera exactly like training
    coords = j3d.reshape(-1,3)
    R_body = np.linalg.norm(coords,axis=1).max()
    min_z   = coords[:,2].min()
    min_xy  = R_body * (1.5/2)
    while True:
        v = sample_direction()
        d = random.uniform(1.5, 2.5)
        cam_pos = v * R_body * d
        if cam_pos[2] < min_z or np.linalg.norm(cam_pos[:2]) < min_xy:
            continue
        R_w2c = look_at_R(cam_pos)
        t_w2c = - R_w2c.dot(cam_pos)
        break

    f  = random.uniform(F_MIN, F_MAX)
    cx = IMG_W/2 + np.random.uniform(-0.03,0.03)*IMG_W
    cy = IMG_H/2 + np.random.uniform(-0.03,0.03)*IMG_H
    K_pix = np.array([[f,0,cx],[0,f,cy],[0,0,1]], np.float32)

    # segment lengths from first frame
    j0 = j3d[0]
    segs = np.array([np.linalg.norm(j0[i]-j0[j]) for i,j in SEG_PAIRS],
                    dtype=np.float32) / SEG_NORM

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = TransformerLifter().to(device)
    ckpt   = torch.load(args.model, map_location=device)
    sd     = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd)

    # inference
    print("Running sliding-window inference...")
    pred_j3d = predict_traj(
        j3d, model, device, args.window,
        K_pix, R_w2c, t_w2c, segs
    )

    # MPJPE
    err = compute_mpjpe(j3d, pred_j3d)
    print(f"MPJPE: {err:.4f} m")

    # save
    np.savez_compressed(args.out,
                        gt_joints   = j3d,
                        pred_joints = pred_j3d)
    print("Results saved to", args.out)


if __name__ == '__main__':
    main()
