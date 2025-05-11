#!/usr/bin/env python3
"""
evaluate_models.py

Evaluate the latest checkpoint for a given model_type.
If model_type == "versatile", runs four masking modes:
  - full:     no masking
  - cam_only: mask segments only
  - seg_only: mask camera intrinsics only
  - none:     mask both
Prints MPJPE per mode and a summary.
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
    """
    Sliding-window inference on j3d (F,J,3). Returns (F,J,3).
    mask_cam: zero out camera token
    mask_seg: zero out segment token
    """
    F,J,_ = j3d.shape
    pred_acc = np.zeros((F,J,3), dtype=np.float32)
    count_acc= np.zeros((F,1,1), dtype=np.float32)

    # build tokens
    if mask_cam:
        f_norm = cx_norm = cy_norm = 0.0
    else:
        f_norm  = K[0,0] / 2000.0
        cx_norm = (K[0,2] - IMG_W/2)/IMG_W
        cy_norm = (K[1,2] - IMG_H/2)/IMG_H
    k_vec = torch.tensor([f_norm, cx_norm, cy_norm],
                         dtype=torch.float32, device=device).unsqueeze(0)

    if mask_seg:
        seg_tok_vec = np.zeros_like(seg_vec, dtype=np.float32)
    else:
        seg_tok_vec = seg_vec
    seg_t = torch.from_numpy(seg_tok_vec).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        for start in range(0, F - window + 1):
            end = start + window
            win3d = j3d[start:end]             # (T,J,3)
            X = win3d.reshape(-1,3).T          # (3,T*J)
            Xc = R_w2c @ X + t_w2c[:,None]     # (3,T*J)
            uvw= K @ Xc                        # (3,T*J)
            uv = (uvw[:2]/(uvw[2:]+1e-8)).T     # (T*J,2)
            uv = uv.reshape(window, J, 2) / IMG_W

            x2d = torch.from_numpy(uv.astype(np.float32))\
                       .unsqueeze(0).to(device)   # (1,T,J,2)
            pred3d = model(x2d, k_vec, seg_t)  # (1,T,J,3)
            pred   = pred3d[0].cpu().numpy()   # (T,J,3)

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
        t_w2c = - R_w2c.dot(cam_pos)
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
    p = argparse.ArgumentParser()
    p.add_argument('--models_dir', required=True)
    p.add_argument('--npz_dir',    required=True)
    p.add_argument('--model_type', required=True,
                   choices=['full','seg','cam','2D','versatile'])
    p.add_argument('--window',     type=int, default=13)
    p.add_argument('--seed',       type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # pick only the highest‐epoch checkpoint
    pat = f"best_lifter_{args.model_type}_128_ddp_epoch*.pt"
    all_ck = sorted(glob.glob(os.path.join(args.models_dir, pat)))
    if not all_ck:
        print(f"[ERROR] no checkpoints for {pat}", file=sys.stderr)
        sys.exit(1)
    ckpt = all_ck[-1]
    print(f"[INFO] Evaluating checkpoint: {ckpt}\n")

    npz_files = sorted(glob.glob(os.path.join(args.npz_dir, '*.npz')))
    if not npz_files:
        print(f"[ERROR] no .npz in {args.npz_dir}", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] {len(npz_files)} validation clips\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO] device = {device}\n")

    # load model
    model = TransformerLifter().to(device)
    ck = torch.load(ckpt, map_location=device)
    sd = ck.get('model_state_dict', ck)
    model.load_state_dict(sd)
    epoch = int(os.path.basename(ckpt).split('epoch')[-1].split('.pt')[0])

    # define modes
    if args.model_type == 'versatile':
        modes = {
            'full':     (False, False),
            'cam_only': (False, True),
            'seg_only': (True,  False),
            'none':     (True,  True),
        }
    else:
        modes = {'full': (False, False)}

    results = []
    # evaluate each mode
    for mode, (mask_cam, mask_seg) in modes.items():
        mpjpes = []
        desc = f"Mode={mode}"
        for fn in tqdm(npz_files, desc=desc, unit="clip"):
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
            mpjpes.append(compute_mpjpe(j3d, pred))

        mean_err = float(np.mean(mpjpes))
        print(f"[RESULT] Epoch {epoch:3d}, mode={mode:8s} → MPJPE = {mean_err:.4f} m")
        results.append((mode, mean_err))

    # summary
    best_mode, best_err = min(results, key=lambda x: x[1])
    print("\n[SUMMARY]")
    for mode, err in results:
        mark = " <-- best" if mode == best_mode else ""
        print(f"  mode={mode:8s}: {err:.4f} m{mark}")
    print(f"\n>>> Best under mode={best_mode}: MPJPE={best_err:.4f} m")

if __name__ == '__main__':
    main()
