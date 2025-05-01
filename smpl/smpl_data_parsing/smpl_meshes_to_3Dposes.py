import os
import numpy as np
import torch
import pandas as pd
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

# Configuration
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir       = os.path.join(project_root, 'amass', 'smplh', 'raw_npz_data')
out_dir       = os.path.join(project_root, 'amass', 'smplh', 'joint_segment_data')
marker_csv    = os.path.join(project_root, 'amass', 'vertices_keypoints_corr.csv')
bm_base       = os.path.join(project_root, 'models', 'amass', 'smplx')

# Make sure output directory exists
os.makedirs(out_dir, exist_ok=True)

# Load marker definitions once
markers_df = pd.read_csv(marker_csv)
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

def compute_and_save(npz_path):
    # derive motion_name from filename
    motion_name = os.path.splitext(os.path.basename(npz_path))[0]
    print(f"Processing {motion_name}...")

    # 1. Load AMASS
    bdata = np.load(npz_path)
    gender = bdata['gender'].item()
    if isinstance(gender, bytes):
        gender = gender.decode('utf-8')

    # 2. Build BodyModel inputs
    body_parms = {
        'root_orient': torch.tensor(bdata['poses'][:, :3], dtype=torch.float32).to(comp_device),
        'pose_body'  : torch.tensor(bdata['poses'][:, 3:66], dtype=torch.float32).to(comp_device),
        'pose_hand'  : torch.tensor(bdata['poses'][:, 66:], dtype=torch.float32).to(comp_device),
        'trans'      : torch.tensor(bdata['trans'],     dtype=torch.float32).to(comp_device),
    }

    # 3. Instantiate SMPL-X
    bm_path = os.path.join(bm_base, gender, 'model.npz')
    bm = BodyModel(bm_fname=bm_path, num_betas=16).to(comp_device)
    body = bm(**body_parms)
    vertices = c2c(body.v)   # (num_frames, num_vertices, 3)

    num_frames = vertices.shape[0]

    # 4. Extract raw marker trajectories
    marker_trajs = {
        name: vertices[:, idx, :]
        for name, idx in marker_list
    }

    # 5. Compute joint centres
    data = {'Frame': np.arange(num_frames)}
    joints = {}
    for jname, mlist in joint_defs.items():
        arrs = [marker_trajs[m] for m in mlist]
        traj = np.stack(arrs, axis=0).mean(axis=0)  # (num_frames,3)
        joints[jname] = traj
        data[f'{jname}_x'] = traj[:,0]
        data[f'{jname}_y'] = traj[:,1]
        data[f'{jname}_z'] = traj[:,2]

    hip_mid = (joints['r_hip'] + joints['l_hip']) / 2  # (num_frames,3)
    for jname, traj in joints.items():
        # subtract hip_mid from each joint trajectory
        centered = traj - hip_mid
        joints[jname] = centered
        # store into your data dict:
        data[f'{jname}_x'] = centered[:, 0]
        data[f'{jname}_y'] = centered[:, 1]
        data[f'{jname}_z'] = centered[:, 2]

    # 6. Compute mean segment lengths
    def dist(a, b):
        return np.linalg.norm(a - b, axis=1)

    fa_r = dist(joints['r_wrist'],   joints['r_elbow'])
    fa_l = dist(joints['l_wrist'],   joints['l_elbow'])
    forearm     = np.concatenate([fa_r, fa_l]).mean()

    ua_r = dist(joints['r_shoulder'], joints['r_elbow'])
    ua_l = dist(joints['l_shoulder'], joints['l_elbow'])
    upperarm    = np.concatenate([ua_r, ua_l]).mean()

    bi_acromial = dist(joints['r_shoulder'], joints['l_shoulder']).mean()

    shoulder_mid = (joints['r_shoulder'] + joints['l_shoulder']) / 2
    hip_mid      = (joints['r_hip']      + joints['l_hip'])      / 2
    spine       = dist(shoulder_mid, hip_mid).mean()

    th_r   = dist(joints['r_hip'],  joints['r_knee'])
    th_l   = dist(joints['l_hip'],  joints['l_knee'])
    thigh  = np.concatenate([th_r, th_l]).mean()

    sh_r   = dist(joints['r_knee'], joints['r_ankle'])
    sh_l   = dist(joints['l_knee'], joints['l_ankle'])
    shank  = np.concatenate([sh_r, sh_l]).mean()

    # 7. Fill constant columns
    consts = {
        'forearm': forearm,
        'upperarm': upperarm,
        'bi_acromial': bi_acromial,
        'spine': spine,
        'thigh': thigh,
        'shank': shank
    }
    for name, val in consts.items():
        data[name] = np.full(num_frames, val)

    # 8. Save to CSV
    df = pd.DataFrame(data)
    out_csv = os.path.join(out_dir, f'{motion_name}_joints_segments.csv')
    df.to_csv(out_csv, index=False)
    print(f" → Saved to {out_csv}")

# Main loop
for fname in sorted(os.listdir(raw_dir)):
    if fname.lower().endswith('.npz'):
        compute_and_save(os.path.join(raw_dir, fname))


