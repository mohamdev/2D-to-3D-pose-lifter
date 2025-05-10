#!/usr/bin/env python3
"""
evaluate_precomputed.py

Load a .npz with keys 
  'joints_3d'        (F,J,3),
  'joints_2d'        (P,F,J,2),
  'segments_lengths' (F,S,1),
  'K'                (P,3,3),
  'R'                (P,3,3),
  't'                (P,3),
evaluate the first --ncams views (or a random subset), compute MPJPE,
and save out a new .npz with ground truth and predicted 3D trajectories.
"""

import sys
import argparse
import random

import numpy as np
import torch

from train_lifter_ddp_dynamic_cams_tiny import TransformerLifter
import train_lifter_ddp_dynamic_cams_tiny as dyn  # for IMG_W, etc.


def compute_mpjpe(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(gt - pred, axis=-1)))


def predict_for_cam(data, model, device, cam_idx, window):
    """
    Exactly your old predictor, loading 2D/intrinsics from data.
    Returns (F,J,3).
    """
    joints_2d = data['joints_2d'][cam_idx]          # (F,J,2)
    segs      = data['segments_lengths'][:,:,0]     # (F,S)
    K_pix     = data['K'][cam_idx]                  # (3,3)
    F, J, _   = joints_2d.shape

    # normalize intrinsics
    f_norm  = K_pix[0,0] / 2000.0
    cx_norm = (K_pix[0,2] - dyn.IMG_W/2) / dyn.IMG_W
    cy_norm = (K_pix[1,2] - dyn.IMG_H/2) / dyn.IMG_H
    k_vec   = torch.tensor([f_norm, cx_norm, cy_norm],
                           dtype=torch.float32, device=device).unsqueeze(0)

    pred_acc  = np.zeros((F, J, 3), dtype=np.float32)
    count_acc = np.zeros((F,1,1), dtype=np.float32)

    model.eval()
    with torch.no_grad():
        for start in range(0, F - window + 1):
            end     = start + window
            x2d_win = joints_2d[start:end]               # (T,J,2)
            seg_vec = segs[start].astype(np.float32)/2.0  # (S,)

            x2d = torch.from_numpy((x2d_win.astype(np.float32)/dyn.IMG_W))\
                       .unsqueeze(0).to(device)          # (1,T,J,2)
            seg = torch.from_numpy(seg_vec).unsqueeze(0).to(device)  # (1,S)

            pred3d = model(x2d, k_vec, seg)               # (1,T,J,3)
            pred_np = pred3d[0].cpu().numpy()             # (T,J,3)

            pred_acc[start:end] += pred_np
            count_acc[start:end] += 1.0

    return (pred_acc / count_acc).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz',    required=True,
                        help="Input .npz with precomputed 2D, intrinsics, etc.")
    parser.add_argument('--model',  required=True,
                        help="TransformerLifter checkpoint")
    parser.add_argument('--window', type=int, default=13,
                        help="Sliding window size")
    parser.add_argument('--ncams',  type=int, default=1,
                        help="Number of cameras to evaluate (default=1)")
    parser.add_argument('--seed',   type=int, default=0,
                        help="Random seed for camera selection")
    parser.add_argument('--out',    required=True,
                        help="Output .npz path")
    args = parser.parse_args()

    # reproducibility
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data = np.load(args.npz)
    # required keys
    for k in ('joints_3d','joints_2d','segments_lengths','K','R','t'):
        if k not in data:
            print(f"Error: '{k}' missing from {args.npz}")
            sys.exit(1)

    gt_j3d = data['joints_3d']            # (F,J,3)
    P, F, J, _ = data['joints_2d'].shape

    # pick camera indices
    all_idxs = list(range(P))
    if args.ncams >= P:
        cam_idxs = all_idxs
    else:
        cam_idxs = random.sample(all_idxs, args.ncams)

    print(f"Evaluating cameras: {cam_idxs}")

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = TransformerLifter().to(device)
    ckpt   = torch.load(args.model, map_location=device)
    sd     = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd)
    print("Model loaded on", device)

    # inference for each cam
    preds = []
    errors = []
    for c in cam_idxs:
        print(f"> Camera {c} …", end=' ')
        pred3d = predict_for_cam(data, model, device, c, args.window)
        err = compute_mpjpe(gt_j3d, pred3d)
        print(f"MPJPE = {err:.4f} m")
        preds.append(pred3d)
        errors.append(err)

    mean_err = float(np.mean(errors))
    print(f"\nMean MPJPE over {len(cam_idxs)} cams: {mean_err:.4f} m")

    # save out
    preds = np.stack(preds, axis=0)  # (C,F,J,3)
    np.savez_compressed(args.out,
                        gt_joints    = gt_j3d,
                        pred_joints  = preds,
                        cam_indices  = np.array(cam_idxs, dtype=np.int32))
    print("Saved:", args.out)


if __name__ == '__main__':
    main()
