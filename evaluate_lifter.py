#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import pandas as pd
import torch
from glob import glob
from tqdm import tqdm
from typing import Tuple

# import your model and dataset classes & loss
from train_lifter_ddp import TransformerLifter, mpjpe

JOINT_NAMES = [
    'r_shoulder','l_shoulder',
    'r_elbow',   'l_elbow',
    'r_wrist',   'l_wrist',
    'r_hip',     'l_hip',
    'r_knee',    'l_knee',
    'r_ankle',   'l_ankle',
]

IMG_W, IMG_H = 1920, 1080
T = 13

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

def plot_all_joint_trajectories(gt_traj, pred_traj, joint_names):
    """
    For each joint, plots X/Y/Z over frames for GT vs Pred.
    
    gt_traj: (F, J, 3) numpy array
    pred_traj: (F, J, 3) numpy array
    joint_names: list of length J
    """
    F, J, _ = gt_traj.shape
    frames = np.arange(F)
    coords = ['X', 'Y', 'Z']
    
    for j in range(J):
        name = joint_names[j]
        fig, axes = plt.subplots(3, 1, figsize=(8, 12))
        
        for i in range(3):
            axes[i].plot(frames, gt_traj[:, j, i], label='GT')
            axes[i].plot(frames, pred_traj[:, j, i], label='Pred')
            axes[i].set_ylabel(f'{coords[i]} position (m)')
            axes[i].legend()
        
        axes[2].set_xlabel('Frame')
        fig.suptitle(f'Joint Trajectory: {name}')
        plt.tight_layout()
        plt.show()

    # Example usage after evaluating one .npz:
    # gt_traj, pred_traj, _, _ = predict_full_trajectory(data, model, device)
    # plot_all_joint_trajectories(gt_traj, pred_traj, JOINT_NAMES)


def predict_full_trajectory(data, model, device):
    """
    Given loaded npz data for one take (multiple cameras),
    runs the model over sliding windows on each camera view and
    reconstructs the full (F, J, 3) predicted trajectory by averaging
    overlapping windows.
    Returns:
      gt_traj:   (F, J, 3) numpy array
      pred_traj: (F, J, 3) numpy array
      times:     list of inference times (s) per window
      errors:    list of MPJPE (m) per window
    """
    joints_2d = data['joints_2d']   # (P, F, J, 2)
    segs      = data['segments_lengths']  # (F, S, 1)
    Ks        = data['K']           # (P, 3, 3)
    P, F, J, _ = joints_2d.shape

    # prepare accumulators
    pred_acc  = np.zeros((F, J, 3), dtype=np.float32)
    count_acc = np.zeros((F,1,1), dtype=np.float32)
    times, errors = [], []

    model.eval()
    with torch.no_grad():
        for p in range(P):
            # normalize intrinsics once
            K_pix = Ks[p]
            f_norm  = K_pix[0,0] / 2000.0
            cx_norm = (K_pix[0,2] - IMG_W/2)/IMG_W
            cy_norm = (K_pix[1,2] - IMG_H/2)/IMG_H
            k_vec = torch.tensor([f_norm, cx_norm, cy_norm],
                                 dtype=torch.float32, device=device)

            for start in range(0, F - T + 1):
                end = start + T

                x2d_win = joints_2d[p, start:end]        # (T, J, 2)
                seg_vec  = segs[start].squeeze().astype(np.float32)/2.0  # (S,)

                # to torch
                x2d = torch.from_numpy(x2d_win.astype(np.float32)/IMG_W)\
                             .unsqueeze(0).to(device)       # (1,T,J,2)
                k    = k_vec.unsqueeze(0)                  # (1,3)
                seg  = torch.from_numpy(seg_vec).unsqueeze(0).to(device)  # (1,S)

                # inference + timing
                start_t = time.time()
                pred3d = model(x2d, k, seg)  # (1,T,J,3)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append(time.time() - start_t)

                pred_np = pred3d[0].cpu().numpy()  # (T,J,3)
                gt_np   = data['joints_3d'][start:end]  # (T,J,3)

                # error
                errors.append(float(mpjpe(pred3d, torch.from_numpy(gt_np).unsqueeze(0).to(device)).item()))

                # accumulate
                pred_acc[start:end] += pred_np
                count_acc[start:end] += 1.0
            print("mean error:", np.mean(np.array(errors)))
    # average overlapping windows
    pred_traj = pred_acc / count_acc
    gt_traj   = data['joints_3d']  # (F,J,3)
    return gt_traj, pred_traj, times, errors

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load model
    model      = TransformerLifter().to(device)
    checkpoint = torch.load(args.model, map_location=device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)

    # iterate over all npz files in data folder
    rows = []
    all_times, all_errors = [], []
    sample_idx = 0

    for npz_path in sorted(glob(os.path.join(args.data, '*.npz'))):
        data = np.load(npz_path)
        gt_traj, pred_traj, times, errors = predict_full_trajectory(data, model, device)

        # plot_all_joint_trajectories(gt_traj, pred_traj, JOINT_NAMES)
        F, J, _ = gt_traj.shape
        # record
        for f in range(F):
            for j in range(J):
                rows.append({
                    'sample': sample_idx,
                    'frame' : f,
                    'joint' : JOINT_NAMES[j],
                    'gt_x'   : float(gt_traj[f,j,0]),
                    'gt_y'   : float(gt_traj[f,j,1]),
                    'gt_z'   : float(gt_traj[f,j,2]),
                    'pred_x' : float(pred_traj[f,j,0]),
                    'pred_y' : float(pred_traj[f,j,1]),
                    'pred_z' : float(pred_traj[f,j,2]),
                })
        all_times.extend(times)
        all_errors.extend(errors)
        sample_idx += 1

    # report
    mean_err = np.mean(all_errors)
    avg_time_ms = np.mean(all_times) * 1000.0
    print(f"\nMean MPJPE over {len(all_errors)} windows: {mean_err:.4f} m")
    print(f"Average inference time per window: {avg_time_ms:.3f} ms")

    # write CSV
    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f"Results written to {args.output}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--model',  required=True,
                   help="Path to best_lifter.pt")
    p.add_argument('--data',   required=True,
                   help="Directory of .npz files")
    p.add_argument('--output', default="eval_fulltraj.csv",
                   help="CSV output filename")
    args = p.parse_args()
    evaluate(args)
