import os
import numpy as np
import torch
import pandas as pd
import traceback
from human_body_prior.body_model.body_model import BodyModel
from human_body_prior.tools.omni_tools import copy2cpu as c2c

# Configuration
comp_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
raw_dir    = os.path.join(project_root, 'amass', 'smplh', 'raw_npz_data')
out_dir    = os.path.join(project_root, 'amass', 'smplh', 'joint_segment_data')
marker_csv = os.path.join(project_root, 'amass', 'vertices_keypoints_corr.csv')
bm_base    = os.path.join(project_root, 'models', 'amass', 'smplx')

os.makedirs(out_dir, exist_ok=True)

# Load marker definitions
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

def compute_and_save(npz_path: str, file_id: int):
    """Load a single .npz, compute joints & segments, save CSV with a file_id column."""
    motion_name = os.path.splitext(os.path.basename(npz_path))[0]
    print(f"Processing [{file_id}] {motion_name}…")
    try:
        # 1) Load AMASS data
        bdata = np.load(npz_path)
        gender = bdata['gender'].item()
        if isinstance(gender, bytes):
            gender = gender.decode('utf-8')

        # 2) Prepare BodyModel inputs
        body_parms = {
            'root_orient': torch.tensor(bdata['poses'][:, :3],   dtype=torch.float32).to(comp_device),
            'pose_body'  : torch.tensor(bdata['poses'][:, 3:66], dtype=torch.float32).to(comp_device),
            'pose_hand'  : torch.tensor(bdata['poses'][:, 66:],  dtype=torch.float32).to(comp_device),
            'trans'      : torch.tensor(bdata['trans'],         dtype=torch.float32).to(comp_device),
        }

        # 3) Instantiate SMPL-X and get vertices
        bm_path = os.path.join(bm_base, gender, 'model.npz')
        bm      = BodyModel(bm_fname=bm_path, num_betas=16).to(comp_device)
        body    = bm(**body_parms)
        vertices = c2c(body.v)
        num_frames = vertices.shape[0]

        # 4) Extract raw marker trajectories
        marker_trajs = {name: vertices[:, idx, :] for name, idx in marker_list}

        # 5) Compute joint centers and center them on the hip midpoint
        data = {
            'file_id': np.full(num_frames, file_id, dtype=int),
            'Frame'  : np.arange(num_frames, dtype=int)
        }
        joints = {}
        for jname, mlist in joint_defs.items():
            stacks = [marker_trajs[m] for m in mlist]
            joint_pos = np.stack(stacks, axis=0).mean(axis=0)
            joints[jname] = joint_pos

        hip_mid = (joints['r_hip'] + joints['l_hip']) / 2.0
        for jname, traj in joints.items():
            centered = traj - hip_mid
            data[f'{jname}_x'] = centered[:, 0]
            data[f'{jname}_y'] = centered[:, 1]
            data[f'{jname}_z'] = centered[:, 2]
            joints[jname] = centered

        # 6) Compute mean segment lengths
        def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            return np.linalg.norm(a - b, axis=1)

        fa_r = dist(joints['r_wrist'],   joints['r_elbow'])
        fa_l = dist(joints['l_wrist'],   joints['l_elbow'])
        forearm = np.concatenate([fa_r, fa_l]).mean()

        ua_r = dist(joints['r_shoulder'], joints['r_elbow'])
        ua_l = dist(joints['l_shoulder'], joints['l_elbow'])
        upperarm = np.concatenate([ua_r, ua_l]).mean()

        bi_acromial = dist(joints['r_shoulder'], joints['l_shoulder']).mean()

        shoulder_mid = (joints['r_shoulder'] + joints['l_shoulder']) / 2.0
        hip_mid2     = (joints['r_hip']      + joints['l_hip'])      / 2.0
        spine        = dist(shoulder_mid, hip_mid2).mean()

        th_r   = dist(joints['r_hip'],  joints['r_knee'])
        th_l   = dist(joints['l_hip'],  joints['l_knee'])
        thigh  = np.concatenate([th_r, th_l]).mean()

        sh_r   = dist(joints['r_knee'], joints['r_ankle'])
        sh_l   = dist(joints['l_knee'], joints['l_ankle'])
        shank  = np.concatenate([sh_r, sh_l]).mean()

        consts = {
            'forearm'    : forearm,
            'upperarm'   : upperarm,
            'bi_acromial': bi_acromial,
            'spine'      : spine,
            'thigh'      : thigh,
            'shank'      : shank
        }
        for name, val in consts.items():
            data[name] = np.full(num_frames, val, dtype=float)

        # 7) Save to CSV
        df      = pd.DataFrame(data)
        out_csv = os.path.join(out_dir, f'{file_id}_{motion_name}_joints_segments.csv')
        df.to_csv(out_csv, index=False)
        print(f" → Saved to {out_csv}")

    except Exception as e:
        print(f" → Failed [{file_id}] {motion_name}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    # Gather all .npz files recursively
    all_npz_paths = []
    for root, dirs, files in os.walk(raw_dir):
        for fname in files:
            if fname.lower().endswith('.npz'):
                all_npz_paths.append(os.path.join(root, fname))

    all_npz_paths.sort()

    # Process each, continuing on error
    for idx, npz_path in enumerate(all_npz_paths, start=1):
        motion_name = os.path.splitext(os.path.basename(npz_path))[0]
        out_csv = os.path.join(
            out_dir,
            f"{idx}_{motion_name}_joints_segments.csv"
        )
        if os.path.exists(out_csv):
            print(f"Skipping [{idx}] {motion_name}, already processed.")
            continue

        compute_and_save(npz_path, file_id=idx)
