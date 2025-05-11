#!/usr/bin/env python3
"""
evaluate_tasks.py

Evaluate the latest checkpoint for a given model_type across multiple "tasks".
Each task is a subfolder under --tasks_dir, containing .npz clips.
For each task (and for each masking mode if model_type=="versatile"), computes:
  - mean MPJPE ± std MPJPE over that task's clips
Then computes overall mean ± std across all clips (within each mode).
Prints a per-task table and overall summary.
"""
import os
import sys
import glob
import argparse
import random
import numpy as np
import torch
from tqdm import tqdm

# import exactly the same definitions you used for training:
from train_lifter_ddp_dynamic_cams_tiny import (
    TransformerLifter,
    sample_direction,
    look_at_R,
    IMG_W, IMG_H, F_MIN, F_MAX, D_MAX, MARGIN, SEG_NORM
)

# segment pairs as in your DynamicPoseDataset
SEG_PAIRS = [(2,4),(3,5),(0,1),(0,6),(6,8),(7,9)]

def compute_mpjpe(gt: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.linalg.norm(gt - pred, axis=-1)))

def predict_traj(
    j3d: np.ndarray,
    model: torch.nn.Module,
    device: torch.device,
    window: int,
    K: np.ndarray,
    R_w2c: np.ndarray,
    t_w2c: np.ndarray,
    seg_vec: np.ndarray,
    mask_cam: bool = False,
    mask_seg: bool = False
) -> np.ndarray:
    F,J,_ = j3d.shape
    pred_acc  = np.zeros((F,J,3), dtype=np.float32)
    count_acc = np.zeros((F,1,1), dtype=np.float32)

    # camera token
    if mask_cam:
        f_norm = cx_norm = cy_norm = 0.0
    else:
        f_norm  = K[0,0] / 2000.0
        cx_norm = (K[0,2] - IMG_W/2)/IMG_W
        cy_norm = (K[1,2] - IMG_H/2)/IMG_H
    k_vec = torch.tensor([f_norm, cx_norm, cy_norm],
                         dtype=torch.float32, device=device).unsqueeze(0)

    # segment token
    if mask_seg:
        seg_tok_vec = np.zeros_like(seg_vec, dtype=np.float32)
    else:
        seg_tok_vec = seg_vec
    seg_t = torch.from_numpy(seg_tok_vec).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        for start in range(0, F - window + 1):
            end = start + window
            win3d = j3d[start:end]               # (T,J,3)
            X = win3d.reshape(-1,3).T            # (3,T*J)
            Xc = R_w2c @ X + t_w2c[:,None]       # (3,T*J)
            uvw = K @ Xc                         # (3,T*J)
            uv  = (uvw[:2]/(uvw[2:]+1e-8)).T      # (T*J,2)
            uv  = uv.reshape(window, J, 2) / IMG_W

            x2d = torch.from_numpy(uv.astype(np.float32))\
                       .unsqueeze(0).to(device)   # (1,T,J,2)
            pred3d = model(x2d, k_vec, seg_t)    # (1,T,J,3)
            pred   = pred3d[0].cpu().numpy()     # (T,J,3)

            pred_acc[start:end] += pred
            count_acc[start:end] += 1.0

    return (pred_acc / count_acc).astype(np.float32)

def sample_camera_and_segs(j3d: np.ndarray):
    coords = j3d.reshape(-1,3)
    R_body = np.linalg.norm(coords,axis=1).max()
    min_z   = coords[:,2].min()
    min_xy  = R_body * (1.5/2)
    # extrinsic
    while True:
        v = sample_direction()
        d = random.uniform(1.5, 2.5)
        cam_pos = v * R_body * d
        if cam_pos[2] < min_z or np.linalg.norm(cam_pos[:2]) < min_xy:
            continue
        R_w2c = look_at_R(cam_pos)
        t_w2c = -R_w2c.dot(cam_pos)
        break
    # intrinsics
    f  = random.uniform(F_MIN, F_MAX)
    cx = IMG_W/2 + np.random.uniform(-0.03,0.03)*IMG_W
    cy = IMG_H/2 + np.random.uniform(-0.03,0.03)*IMG_H
    K_pix = np.array([[f,0,cx],[0,f,cy],[0,0,1]], np.float32)
    # segments
    j0 = j3d[0]
    segs = np.array([np.linalg.norm(j0[i]-j0[j]) for i,j in SEG_PAIRS],
                    dtype=np.float32) / SEG_NORM
    return K_pix, R_w2c, t_w2c, segs

def main():
    p = argparse.ArgumentParser(
        description="Evaluate latest versatile/full/seg/cam/2D model per task"
    )
    p.add_argument('--models_dir', required=True,
                   help="Directory with best_lifter_{model_type}_...pt")
    p.add_argument('--tasks_dir',  required=True,
                   help="Directory containing subfolders for each task")
    p.add_argument('--model_type', required=True,
                   choices=['full','seg','cam','2D','versatile'])
    p.add_argument('--window',     type=int, default=13,
                   help="Sliding window size (frames)")
    p.add_argument('--seed',       type=int, default=0,
                   help="Random seed for camera sampling")
    args = p.parse_args()

    # reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # pick latest checkpoint
    pat = f"best_lifter_{args.model_type}_128_ddp_epoch*.pt"
    all_ck = sorted(glob.glob(os.path.join(args.models_dir, pat)))
    if not all_ck:
        print(f"[ERROR] no checkpoints for {pat}", file=sys.stderr)
        sys.exit(1)
    ckpt = all_ck[-1]
    epoch = int(os.path.basename(ckpt).split('epoch')[-1].split('.pt')[0])
    print(f"[INFO] Evaluating checkpoint: {ckpt} (epoch {epoch})\n")

    # determine modes
    if args.model_type == 'versatile':
        modes = {
            'full':     (False, False),
            'cam_only': (False, True),
            'seg_only': (True,  False),
            'none':     (True,  True),
        }
    else:
        modes = {
            'full':     (False, False),
            'cam_only': (False, True),
            'seg_only': (True,  False),
            'none':     (True,  True),
        }

    # load model once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerLifter().to(device)
    ck = torch.load(ckpt, map_location=device)
    sd = ck.get('model_state_dict', ck)
    model.load_state_dict(sd)

    # find tasks
    task_names = sorted([
        d for d in os.listdir(args.tasks_dir)
        if os.path.isdir(os.path.join(args.tasks_dir, d))
    ])
    if not task_names:
        print(f"[ERROR] no task subfolders in {args.tasks_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] Found tasks: {', '.join(task_names)}\n")

    all_results = {mode: {} for mode in modes}
    overall_clip_errors = {mode: [] for mode in modes}

    # iterate tasks
    for task in task_names:
        task_folder = os.path.join(args.tasks_dir, task)
        npz_files = sorted(glob.glob(os.path.join(task_folder, '*.npz')))
        if not npz_files:
            print(f"[WARN] no .npz in task '{task}' – skipping", file=sys.stderr)
            continue
        print(f"[TASK] '{task}' – {len(npz_files)} clips")

        # sample each clip once (reset seeds)
        for mode, (mask_cam, mask_seg) in modes.items():
            clip_errors = []
            desc = f"{task}:{mode}"
            for fn in tqdm(npz_files, desc=desc, unit="clip", leave=False):
                data = np.load(fn)
                if 'joints_3d' not in data:
                    continue
                j3d = data['joints_3d']
                Kpix, Rw2c, tw2c, segs = sample_camera_and_segs(j3d)
                pred = predict_traj(
                    j3d, model, device, args.window,
                    Kpix, Rw2c, tw2c, segs,
                    mask_cam=mask_cam, mask_seg=mask_seg
                )
                clip_errors.append(compute_mpjpe(j3d, pred))
            if not clip_errors:
                print(f"  [WARN] no valid clips for {task} mode={mode}", file=sys.stderr)
                continue

            mean_err = float(np.mean(clip_errors))
            std_err  = float(np.std(clip_errors))
            all_results[mode][task] = (mean_err, std_err)
            overall_clip_errors[mode].extend(clip_errors)
            print(f"  → mode={mode:8s}: MPJPE = {mean_err:.4f} ± {std_err:.4f} m")
        print("")

    # overall summary
    print("\n[OVERALL SUMMARY]")
    for mode in modes:
        errs = overall_clip_errors[mode]
        if not errs:
            print(f"mode={mode}: no data")
            continue
        om = float(np.mean(errs))
        osd= float(np.std(errs))
        print(f"mode={mode:8s}: MPJPE = {om:.4f} ± {osd:.4f} m")

if __name__ == '__main__':
    main()
