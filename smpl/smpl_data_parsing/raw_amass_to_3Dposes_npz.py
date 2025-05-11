import os
import numpy as np
import torch
import pandas as pd
import traceback
import random
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

# Configuration
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir      = os.path.join(project_root, 'amass', 'smplh', 'raw_npz_data')
out_root     = os.path.join(project_root, 'amass', 'smplh', 'full_amass_3D_poses')

# Split directories
def prepare_dirs(out_root):
    train_dir = os.path.join(out_root, 'train')
    val_dir   = os.path.join(out_root, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    return train_dir, val_dir

# Load marker definitions
marker_csv  = os.path.join(project_root, 'amass', 'vertices_keypoints_corr.csv')
markers_df  = pd.read_csv(marker_csv)
marker_list = [(row['Name'], int(row['Index'])) for _, row in markers_df.iterrows()]

# Joint definitions
joint_defs = {
    'r_shoulder': ['rshoulder'],
    'l_shoulder': ['lshoulder'],
    'r_elbow'   : ['r_lelbow', 'r_melbow'],
    'l_elbow'   : ['l_lelbow', 'l_melbow'],
    'r_wrist'   : ['r_lwrist', 'r_mwrist'],
    'l_wrist'   : ['l_lwrist', 'l_mwrist'],
    'r_hip'     : ['r_ASIS', 'r_PSIS'],
    'l_hip'     : ['l_ASIS', 'l_PSIS'],
    'r_knee'    : ['r_knee', 'r_mknee'],
    'l_knee'    : ['l_knee', 'l_mknee'],
    'r_ankle'   : ['r_ankle', 'r_mankle'],
    'l_ankle'   : ['l_ankle', 'l_mankle'],
}


def compute_and_save(npz_path: str, file_id: int, save_dir: str):
    """Load AMASS .npz, compute canonical 3D joints, save .npz with joints_3d, R, and t."""
    motion_name = os.path.splitext(os.path.basename(npz_path))[0]
    print(f"Processing [{file_id}] {motion_name} → {save_dir}")
    try:
        # 1) Load AMASS data
        bdata = np.load(npz_path)
        gender = bdata['gender'].item()
        if isinstance(gender, bytes): gender = gender.decode('utf-8')

        # 2) Prepare BodyModel inputs
        body_parms = {
            'root_orient': torch.tensor(bdata['poses'][:, :3],   dtype=torch.float32).to(device),
            'pose_body'  : torch.tensor(bdata['poses'][:, 3:66], dtype=torch.float32).to(device),
            'pose_hand'  : torch.tensor(bdata['poses'][:, 66:],  dtype=torch.float32).to(device),
            'trans'      : torch.tensor(bdata['trans'],         dtype=torch.float32).to(device),
        }

        # 3) SMPL-X vertices
        bm_path = os.path.join(project_root, 'models', 'amass', 'smplx', gender, 'model.npz')
        bm      = BodyModel(bm_fname=bm_path, num_betas=16).to(device)
        body    = bm(**body_parms)
        verts   = c2c(body.v)                 # (F, V, 3)

        # 4) Markers → joint centers
        marker_trajs = {name: verts[:, idx, :] for name, idx in marker_list}
        joints = {j: np.stack([marker_trajs[m] for m in ml], axis=0).mean(axis=0)
                  for j, ml in joint_defs.items()}

        # 5a) Center on hip midpoint
        hip_mid = (joints['r_hip'] + joints['l_hip']) * 0.5
        for j in joints: joints[j] -= hip_mid

        # 5b) Build static hip-frame from first frame
        r0 = joints['r_hip'][0]; l0 = joints['l_hip'][0]
        y = (r0 - l0); y /= np.linalg.norm(y) + 1e-8
        sh_mid0 = (joints['r_shoulder'][0] + joints['l_shoulder'][0]) * 0.5
        z = sh_mid0; z /= np.linalg.norm(z) + 1e-8
        x = np.cross(y, z); x /= np.linalg.norm(x) + 1e-8
        z = np.cross(x, y); z /= np.linalg.norm(z) + 1e-8
        R = np.vstack([x, y, z])                              # (3,3)

        # 5c) Rotate into hip frame → joints_3d
        joints_3d = np.stack([joints[j] @ R.T for j in joint_defs.keys()], axis=1)
        # shape: (F, J, 3)

        # Translation of hip frame origin
        t = hip_mid[0]

        # 6) Save .npz
        out_path = os.path.join(save_dir, f"{file_id}_{motion_name}.npz")
        np.savez(out_path, joints_3d=joints_3d, R=R, t=t)
        print(f" → Saved {out_path}")
    except Exception as e:
        print(f" → Failed [{file_id}] {motion_name}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Gather all .npz files
    all_npz = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.lower().endswith('.npz'):
                all_npz.append(os.path.join(root, f))

    # all_npz = all_npz[:10]
    all_npz.sort()

    # Random 80/20 split
    random.shuffle(all_npz)
    N = len(all_npz)
    n_train = int(0.8 * N)
    train_list = all_npz[:n_train]
    val_list   = all_npz[n_train:]

    train_dir, val_dir = prepare_dirs(out_root)
    # Process train
    for idx, path in enumerate(train_list, start=1):
        compute_and_save(path, idx, train_dir)
    # Process val
    for idx, path in enumerate(val_list, start=1):
        compute_and_save(path, idx, val_dir)
