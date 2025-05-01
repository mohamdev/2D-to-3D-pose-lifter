# augment_cameras.py
import os
import numpy as np
import cv2
import pandas as pd

# --- CONFIG ---
N_CAM        = 20             # number of cameras per frame
IMG_W, IMG_H = 1920, 1080     # image size
F_MIN, F_MAX = 800, 1400      # focal length range (px)
D_MAX        = 6.0            # max camera distance (m)
MARGIN       = 1.1            # safety margin on d_min
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_DIR      = os.path.join(PROJECT_ROOT, 'amass', 'smplh', 'joint_segment_data')
OUT_DIR      = os.path.join(CSV_DIR, 'augmented_npz')
os.makedirs(OUT_DIR, exist_ok=True)

# list of joints in the same order as CSV columns
JOINT_NAMES = [
    'r_shoulder','l_shoulder',
    'r_elbow',   'l_elbow',
    'r_wrist',   'l_wrist',
    'r_hip',     'l_hip',
    'r_knee',    'l_knee',
    'r_ankle',   'l_ankle',
]

# --- GEOMETRY HELPERS ---
def fibonacci_sphere(n, randomize=True):
    """Nearly-uniform directions on the unit sphere."""
    rnd = 1.0 if randomize else 0.0
    points = []
    offset = 2.0 / n
    increment = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        y = ((i * offset) - 1) + (offset / 2)
        r = np.sqrt(max(0.0, 1 - y*y))
        phi = ((i + rnd) % n) * increment
        x, z = np.cos(phi)*r, np.sin(phi)*r
        points.append([x, y, z])
    return np.stack(points, 0)  # (n,3)

def look_at_rotation(cam_pos, target=np.zeros(3), up=np.array([0,1,0])):
    """
    Build world->camera rotation R such that camera at cam_pos
    looks at 'target', with given 'up' vector.
    """
    z = target - cam_pos
    z = z / np.linalg.norm(z)
    x = np.cross(up, z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    # rows are the camera axes in world coords
    return np.stack([x, y, z], axis=0)

# --- MAIN PROCESSING ---
for fname in sorted(os.listdir(CSV_DIR)):
    if not fname.endswith('_joints_segments.csv'):
        continue

    motion = fname.replace('_joints_segments.csv','')
    print(f"\n→ Augmenting '{motion}'…")

    # 1. load CSV
    df = pd.read_csv(os.path.join(CSV_DIR, fname))
    F = len(df)
    J = len(JOINT_NAMES)

    # 2. build (F,J,3) array of 3D joints
    joints_3d = np.zeros((F, J, 3), dtype=np.float32)
    for j, name in enumerate(JOINT_NAMES):
        joints_3d[:, j, 0] = df[f'{name}_x']
        joints_3d[:, j, 1] = df[f'{name}_y']
        joints_3d[:, j, 2] = df[f'{name}_z']

    # 3. compute body radius
    R_body = np.max(np.linalg.norm(joints_3d, axis=2))

    # 4. sample camera directions once
    dirs = fibonacci_sphere(N_CAM)

    # prepare storage
    joints_2d = np.zeros((F, N_CAM, J, 2), dtype=np.float32)
    Ks        = np.zeros((N_CAM, 3, 3), dtype=np.float32)
    Rs        = np.zeros((N_CAM, 3, 3), dtype=np.float32)
    ts        = np.zeros((N_CAM, 3   ), dtype=np.float32)
    d_mins    = np.zeros((N_CAM    ), dtype=np.float32)

    # 5. for each camera
    for k, v in enumerate(dirs):
        # intrinsics
        f  = np.random.uniform(F_MIN, F_MAX)
        cx = IMG_W/2 + np.random.uniform(-0.03,0.03)*IMG_W
        cy = IMG_H/2 + np.random.uniform(-0.03,0.03)*IMG_H
        K  = np.array([[f,0,cx],[0,f,cy],[0,0,1]], np.float32)

        # fov & minimal distance
        fov_y    = 2 * np.arctan(IMG_H/(2*f))
        d_min_cam= R_body/np.tan(fov_y/2) * MARGIN

        # choose d in [d_min_cam, D_MAX]
        if d_min_cam > D_MAX:
            d = d_min_cam
        else:
            d = np.random.uniform(d_min_cam, D_MAX)

        cam_pos   = v * d
        R_w2c     = look_at_rotation(cam_pos)          # world->camera
        t_w2c     = -R_w2c.dot(cam_pos)               # translation

        # project each frame
        rvec, _   = cv2.Rodrigues(R_w2c)
        tvec      = t_w2c.reshape(3,1)
        for i in range(F):
            pts, _ = cv2.projectPoints(
                joints_3d[i], rvec, tvec,
                K, distCoeffs=None
            )
            joints_2d[i, k] = pts.reshape(J, 2)

        # store
        Ks[k]     = K
        Rs[k]     = R_w2c
        ts[k]     = t_w2c
        d_mins[k] = d_min_cam

    # 6. save .npz
    out_path = os.path.join(OUT_DIR, f'{motion}_cam{N_CAM}.npz')
    np.savez_compressed(
        out_path,
        joints_3d = joints_3d,
        joints_2d = joints_2d,
        K         = Ks,
        R         = Rs,
        t         = ts,
        d_min     = d_mins
    )
    print(f"   → saved {out_path}")
