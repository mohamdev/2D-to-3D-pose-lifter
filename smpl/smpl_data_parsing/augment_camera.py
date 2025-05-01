#!/usr/bin/env python3
# augment_cameras.py

import os, json
import numpy as np
import cv2
import pandas as pd

# --- USER CONFIG ---
M_EXTR       = 30               # number of distinct extrinsics per clip
L_INTR       = 10               # number of distinct intrinsics per extrinsic
IMG_W, IMG_H = 1920, 1080      # synthetic image resolution
F_MIN, F_MAX = 800, 1400       # focal-length range (px)
D_MAX        = 6.0             # maximum camera distance (m)
MARGIN       = 1.1             # safety margin on d_min
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_DIR = os.path.join(PROJECT_ROOT, 'amass', 'smplh', 'joint_segment_data')
OUT_DIR = os.path.join(PROJECT_ROOT, 'amass', 'smplh', 'augmented_npz_grid')
os.makedirs(OUT_DIR, exist_ok=True)

print("CSV_DIR:", CSV_DIR)
print("OUT_DIR:", OUT_DIR)

# joints order must match your CSV columns
JOINT_NAMES = [
    'r_shoulder','l_shoulder',
    'r_elbow',   'l_elbow',
    'r_wrist',   'l_wrist',
    'r_hip',     'l_hip',
    'r_knee',    'l_knee',
    'r_ankle',   'l_ankle',
]
J = len(JOINT_NAMES)

# segment‐length columns in the CSV, and count
SEG_NAMES = ['forearm','upperarm','bi_acromial','spine','thigh','shank']
S = len(SEG_NAMES)

P = M_EXTR * L_INTR   # total number of takes per clip

# --- HELPERS ---
def fibonacci_dirs(n):
    """
    Nearly‐uniform directions on the unit sphere via
    the Fibonacci lattice.
    Returns array (n,3).
    """
    i = np.arange(n)
    phi = (1 + 5**0.5) / 2
    theta = 2 * np.pi * i / phi
    z = 1 - 2*(i + 0.5)/n
    r = np.sqrt(1 - z*z)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)

def look_at_R(cam_pos, target=np.zeros(3), up=np.array([0,1,0])):
    """
    Build world->camera rotation so that the camera at cam_pos
    looks at 'target', with world-up = up.
    Returns R (3×3).
    """
    z = target - cam_pos
    z = z / np.linalg.norm(z)
    x = np.cross(up, z); x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)  # rows are camera axes in world coords

# --- MAIN ---
for fn in sorted(os.listdir(CSV_DIR)):
    if not fn.endswith('_joints_segments.csv'):
        continue

    motion = fn.replace('_joints_segments.csv','')
    print(f"\n→ Augmenting clip '{motion}' with {P} takes…")

    # 1) load CSV
    print("file_to_read:", os.path.join(CSV_DIR, fn))
    df = pd.read_csv(os.path.join(CSV_DIR, fn))
    
    F = len(df)

    # 2) build (F,J,3) joints_3d
    joints_3d = np.stack([
        df[[f'{n}_x', f'{n}_y', f'{n}_z']].values
        for n in JOINT_NAMES
    ], axis=1).astype(np.float32)  # shape (F,J,3)

    # 3) build (F,S,1) segments_lengths
    segs = np.stack([df[n].values for n in SEG_NAMES], axis=1)  # (F,S)
    segments_lengths = segs.reshape(F, S, 1).astype(np.float32)

    # 4) compute body radius for FoV checks
    R_body = np.linalg.norm(joints_3d.reshape(-1,3), axis=1).max()

    # 5) allocate output arrays
    joints_2d = np.zeros((P, F, J, 2), dtype=np.float32)
    Ks        = np.zeros((P, 3, 3), dtype=np.float32)
    Rs        = np.zeros((P, 3, 3), dtype=np.float32)
    ts        = np.zeros((P, 3   ), dtype=np.float32)
    d_mins    = np.zeros((P   ), dtype=np.float32)

    # 6) sample M extrinsics once
    dirs = fibonacci_dirs(M_EXTR)

    p = 0
    for m in range(M_EXTR):
        v = dirs[m]  # direction

        # for each intrinsic set
        for l in range(L_INTR):
            # sample intrinsics
            f  = np.random.uniform(F_MIN, F_MAX)
            cx = IMG_W/2 + np.random.uniform(-0.03,0.03)*IMG_W
            cy = IMG_H/2 + np.random.uniform(-0.03,0.03)*IMG_H
            K  = np.array([[f,0,cx],[0,f,cy],[0,0,1]], np.float32)

            # compute minimal distance for this f
            fov_y = 2 * np.arctan(IMG_H/(2*f))
            d_min = R_body / np.tan(fov_y/2) * MARGIN
            # sample actual distance
            d = d_min if d_min > D_MAX else np.random.uniform(d_min, D_MAX)

            # build extrinsic
            cam_pos = v * d
            R_w2c   = look_at_R(cam_pos)
            t_w2c   = -R_w2c.dot(cam_pos)

            # project all F×J points in one go
            pts, _ = cv2.projectPoints(
                joints_3d.reshape(-1,3),
                cv2.Rodrigues(R_w2c)[0],
                t_w2c.reshape(3,1),
                K, None
            )
            joints_2d[p] = pts.reshape(F, J, 2)

            # store
            Ks[p]     = K
            Rs[p]     = R_w2c
            ts[p]     = t_w2c
            d_mins[p] = d_min

            p += 1

    # 7) save .npz
    out_path = os.path.join(OUT_DIR, f'{motion}_e{M_EXTR}_i{L_INTR}.npz')
    np.savez_compressed(
        out_path,
        joints_3d        = joints_3d,
        segments_lengths = segments_lengths,
        joints_2d        = joints_2d,
        K                = Ks,
        R                = Rs,
        t                = ts,
        d_min            = d_mins,
        meta             = json.dumps({
            'M_extrinsics': M_EXTR,
            'L_intrinsics': L_INTR,
            'IMG_W': IMG_W,
            'IMG_H': IMG_H
        })
    )
    print(f"   → saved {out_path}")

print("\nDone.")
