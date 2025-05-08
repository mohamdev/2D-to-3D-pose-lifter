#!/usr/bin/env python3
"""
view_cameras_skeleton_with_estimate.py

Load a .npz with keys 'R','t','joints_3d','joints_2d','segments_lengths','K',
display:
  - camera frames (blue)
  - ground-truth skeleton (yellow spheres, green thick lines + spine)
  - estimated skeleton (red spheres, red thick lines + spine)

Estimation is done by loading your TransformerLifter checkpoint and
running sliding-window inference on one chosen camera view.
"""

import sys
import time
import argparse

import numpy as np
import torch
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

from train_lifter_ddp import TransformerLifter

# ----------------------------------------
# configuration
# ----------------------------------------
JOINT_NAMES = [
    'r_shoulder','l_shoulder',
    'r_elbow',   'l_elbow',
    'r_wrist',   'l_wrist',
    'r_hip',     'l_hip',
    'r_knee',    'l_knee',
    'r_ankle',   'l_ankle',
]

CONNECTIONS = [
    ('r_shoulder','r_elbow'),
    ('r_elbow',   'r_wrist'),
    ('l_shoulder','l_elbow'),
    ('l_elbow',   'l_wrist'),
    ('l_shoulder','r_shoulder'),
    ('r_hip',     'r_knee'),
    ('r_knee',    'r_ankle'),
    ('l_hip',     'l_knee'),
    ('l_knee',    'l_ankle'),
    ('l_hip',     'r_hip'),
]

T_DEFAULT = 13  # transformer window size

# colors & sizes
CAMERA_COLOR     = [0,   0,   255, 1]  # blue
GT_JOINT_COLOR   = [1,   1,   0,   1]  # yellow
GT_BONE_COLOR    = [0,   1,   0,   1]  # green
EST_JOINT_COLOR  = [1,   0,   0,   1]  # red
EST_BONE_COLOR   = [1,   0,   0,   1]  # red
BASE_FRAME_COLOR = [255, 0,   0,   1]  # red

JOINT_RAD     = 0.02
CAM_AXIS_RAD  = 0.03
CAM_AXIS_LEN  = 0.08
BASE_AXIS_RAD = 0.02
BASE_AXIS_LEN = 0.15
LINE_WIDTH    = 2.0

# ----------------------------------------
# predictor
# ----------------------------------------
def predict_traj_for_cam(data, model, device, cam_idx, window):
    """
    Sliding-window inference on one camera view.
    Returns pred_traj: (F,J,3) numpy array.
    """
    joints_2d = data['joints_2d'][cam_idx]    # (F,J,2)
    segs      = data['segments_lengths'][:, :, 0]  # (F,S)
    K_pix     = data['K'][cam_idx]             # (3,3)
    F, J2, _  = joints_2d.shape
    assert J2 == len(JOINT_NAMES)
    pred_acc  = np.zeros((F, J2, 3), dtype=np.float32)
    count_acc = np.zeros((F,1,1), dtype=np.float32)

    # normalize intrinsics once
    IMG_W, IMG_H = 1920, 1080
    f_norm  = K_pix[0,0] / 2000.0
    cx_norm = (K_pix[0,2] - IMG_W/2) / IMG_W
    cy_norm = (K_pix[1,2] - IMG_H/2) / IMG_H
    k_vec = torch.tensor([f_norm, cx_norm, cy_norm],
                         dtype=torch.float32, device=device).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        for start in range(0, F - window + 1):
            end = start + window
            x2d_win = joints_2d[start:end]    # (T,J,2)
            seg_vec  = segs[start].astype(np.float32) / 2.0  # (S,)

            x2d = torch.from_numpy(x2d_win.astype(np.float32)/IMG_W)\
                         .unsqueeze(0).to(device)           # (1,T,J,2)
            k   = k_vec                                   # (1,3)
            seg = torch.from_numpy(seg_vec).unsqueeze(0).to(device)  # (1,S)

            pred3d = model(x2d, k, seg)                   # (1,T,J,3)
            pred_np = pred3d[0].cpu().numpy()             # (T,J,3)

            pred_acc[start:end] += pred_np
            count_acc[start:end] += 1.0

    return (pred_acc / count_acc).astype(np.float32)



# ----------------------------------------
# numpy MPJPE
# ----------------------------------------
def compute_mpjpe(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    gt, pred: (F, J, 3)
    returns mean_{f,j} ||gt[f,j] - pred[f,j]||_2
    """
    return float(np.mean(np.linalg.norm(gt - pred, axis=-1)))

# ----------------------------------------
# Gepetto viewer setup
# ----------------------------------------
def gv_init():
    model      = pin.Model()
    geom_model = pin.GeometryModel()
    vis_model  = pin.GeometryModel()
    viz        = GepettoVisualizer(model, geom_model, vis_model)
    try:
        viz.initViewer()
        viz.loadViewerModel("pinocchio")
    except Exception as e:
        print("Error: make sure gepetto-viewer is installed and running.")
        print(e)
        sys.exit(1)

    viz.viewer.gui.addXYZaxis(
        'world/base_frame', BASE_FRAME_COLOR, BASE_AXIS_RAD, BASE_AXIS_LEN
    )
    M0 = pin.SE3(np.eye(3), np.zeros((3,1)))
    viz.viewer.gui.applyConfiguration(
        'world/base_frame', pin.SE3ToXYZQUAT(M0).tolist()
    )
    viz.viewer.gui.refresh()
    return viz

def place_frame(viz, name, M):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())

# ----------------------------------------
# main
# ----------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz',         help="Path to your .npz file")
    p.add_argument('--model',       help="Path to your TransformerLifter checkpoint")
    p.add_argument('--cam',  type=int, default=0,
                   help="Camera index to lift (0..P-1)")
    p.add_argument('--window',type=int, default=T_DEFAULT,
                   help="Temporal window size for the transformer")
    p.add_argument('--allcams',type=int, default=False,
                   help="Temporal window size for the transformer")
    args = p.parse_args()

    # load data
    data = np.load(args.npz)
    Rs         = data['R']           # (P,3,3)
    ts         = data['t']           # (P,3)
    gt_joints  = data['joints_3d']   # (F,J,3)
    P, F, J, _ = data['joints_2d'].shape

    if not (0 <= args.cam < P):
        print(f"Error: camera index must be in [0, {P-1}]")
        sys.exit(1)

    print(f"Data: {P} cameras, {F} frames, {J} joints")

    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = TransformerLifter().to(device)
    ckpt   = torch.load(args.model, map_location=device)
    sd     = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(sd)
    print("TransformerLifter loaded on", device)

    # predict full trajectory for chosen camera
    print("Running sliding-window inference...")
    pred_joints = predict_traj_for_cam(data, model, device, args.cam, args.window)
    print("Done inference.")

    # compute MPJPE
    err = compute_mpjpe(gt_joints, pred_joints)
    print(f"MPJPE (camera {args.cam}): {err:.4f} m")
    
    # init viewer
    viz = gv_init()

    # draw camera frames
    for i, (R_w2c, t_w2c) in enumerate(zip(Rs, ts)):
        if i == args.cam and args.allcams == False: #Only show 
            name = f'world/camera_{i}'
            viz.viewer.gui.addXYZaxis(name, CAMERA_COLOR, CAM_AXIS_RAD, CAM_AXIS_LEN)
            R_c2w = R_w2c.T
            C     = - R_c2w @ t_w2c.reshape(3,1)
            place_frame(viz, name, pin.SE3(R_c2w, C))
        elif args.allcams == True:
            name = f'world/camera_{i}'
            viz.viewer.gui.addXYZaxis(name, CAMERA_COLOR, CAM_AXIS_RAD, CAM_AXIS_LEN)
            R_c2w = R_w2c.T
            C     = - R_c2w @ t_w2c.reshape(3,1)
            place_frame(viz, name, pin.SE3(R_c2w, C))

    # create spheres for GT and EST joints + midpoints
    for jn in JOINT_NAMES + ['mid_hip','mid_shoulder']:
        viz.viewer.gui.addSphere(f'world/{jn}',           JOINT_RAD, GT_JOINT_COLOR)
        viz.viewer.gui.addSphere(f'world/est_{jn}',       JOINT_RAD, EST_JOINT_COLOR)

    # create GT bone lines + spine
    for (a,b) in CONNECTIONS:
        ln = f'world/line_{a}_{b}'
        p1 = gt_joints[0, JOINT_NAMES.index(a)].tolist()
        p2 = gt_joints[0, JOINT_NAMES.index(b)].tolist()
        viz.viewer.gui.addLine(ln, p1, p2, GT_BONE_COLOR)
        viz.viewer.gui.setCurveLineWidth(ln, LINE_WIDTH)
    # GT spine
    mh0 = ((gt_joints[0,JOINT_NAMES.index('r_hip')] +
            gt_joints[0,JOINT_NAMES.index('l_hip')]) / 2.0).tolist()
    ms0 = ((gt_joints[0,JOINT_NAMES.index('r_shoulder')] +
            gt_joints[0,JOINT_NAMES.index('l_shoulder')]) / 2.0).tolist()
    viz.viewer.gui.addLine('world/line_spine', mh0, ms0, GT_BONE_COLOR)
    viz.viewer.gui.setCurveLineWidth('world/line_spine', LINE_WIDTH)

    # create EST bone lines + spine
    for (a,b) in CONNECTIONS:
        ln = f'world/line_est_{a}_{b}'
        p1 = pred_joints[0, JOINT_NAMES.index(a)].tolist()
        p2 = pred_joints[0, JOINT_NAMES.index(b)].tolist()
        viz.viewer.gui.addLine(ln, p1, p2, EST_BONE_COLOR)
        viz.viewer.gui.setCurveLineWidth(ln, LINE_WIDTH)
    # EST spine
    mh0e = ((pred_joints[0,JOINT_NAMES.index('r_hip')] +
             pred_joints[0,JOINT_NAMES.index('l_hip')]) / 2.0).tolist()
    ms0e = ((pred_joints[0,JOINT_NAMES.index('r_shoulder')] +
             pred_joints[0,JOINT_NAMES.index('l_shoulder')]) / 2.0).tolist()
    viz.viewer.gui.addLine('world/line_est_spine', mh0e, ms0e, EST_BONE_COLOR)
    viz.viewer.gui.setCurveLineWidth('world/line_est_spine', LINE_WIDTH)

    viz.viewer.gui.refresh()

    # animation
    try:
        while True:
            for f in range(F):
                # update GT joints
                for j, jn in enumerate(JOINT_NAMES):
                    M = pin.SE3(np.eye(3), np.matrix(gt_joints[f,j]).T)
                    viz.viewer.gui.applyConfiguration(f'world/{jn}',
                                                      pin.SE3ToXYZQUAT(M).tolist())
                # update GT midpoints
                mh = np.mean([gt_joints[f,JOINT_NAMES.index('r_hip')],
                              gt_joints[f,JOINT_NAMES.index('l_hip')]], axis=0)
                ms = np.mean([gt_joints[f,JOINT_NAMES.index('r_shoulder')],
                              gt_joints[f,JOINT_NAMES.index('l_shoulder')]], axis=0)
                viz.viewer.gui.applyConfiguration('world/mid_hip',
                                                  pin.SE3ToXYZQUAT(pin.SE3(np.eye(3), np.matrix(mh).T)).tolist())
                viz.viewer.gui.applyConfiguration('world/mid_shoulder',
                                                  pin.SE3ToXYZQUAT(pin.SE3(np.eye(3), np.matrix(ms).T)).tolist())
                # update GT bones
                for (a,b) in CONNECTIONS:
                    p1 = gt_joints[f,JOINT_NAMES.index(a)].tolist()
                    p2 = gt_joints[f,JOINT_NAMES.index(b)].tolist()
                    viz.viewer.gui.setLineExtremalPoints(f'world/line_{a}_{b}', p1, p2)
                viz.viewer.gui.setLineExtremalPoints('world/line_spine', mh.tolist(), ms.tolist())

                # update EST joints
                for j, jn in enumerate(JOINT_NAMES):
                    M = pin.SE3(np.eye(3), np.matrix(pred_joints[f,j]).T)
                    viz.viewer.gui.applyConfiguration(f'world/est_{jn}',
                                                      pin.SE3ToXYZQUAT(M).tolist())
                # update EST midpoints
                mhe = np.mean([pred_joints[f,JOINT_NAMES.index('r_hip')],
                               pred_joints[f,JOINT_NAMES.index('l_hip')]], axis=0)
                mse = np.mean([pred_joints[f,JOINT_NAMES.index('r_shoulder')],
                               pred_joints[f,JOINT_NAMES.index('l_shoulder')]], axis=0)
                viz.viewer.gui.applyConfiguration('world/est_mid_hip',
                                                  pin.SE3ToXYZQUAT(pin.SE3(np.eye(3), np.matrix(mhe).T)).tolist())
                viz.viewer.gui.applyConfiguration('world/est_mid_shoulder',
                                                  pin.SE3ToXYZQUAT(pin.SE3(np.eye(3), np.matrix(mse).T)).tolist())
                # update EST bones
                for (a,b) in CONNECTIONS:
                    viz.viewer.gui.setLineExtremalPoints(
                        f'world/line_est_{a}_{b}',
                        pred_joints[f,JOINT_NAMES.index(a)].tolist(),
                        pred_joints[f,JOINT_NAMES.index(b)].tolist()
                    )
                viz.viewer.gui.setLineExtremalPoints(
                    'world/line_est_spine',
                    mhe.tolist(), mse.tolist()
                )

                viz.viewer.gui.refresh()
                time.sleep(0.01)  # adjust for desired FPS
    except KeyboardInterrupt:
        print("\nStopped by user. Exiting.")


if __name__ == '__main__':
    main()
