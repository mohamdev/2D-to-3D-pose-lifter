#!/usr/bin/env python3
"""
view_dynamic_npz.py

Load a .npz with 'gt_joints' and 'pred_joints' (F,J,3),
and animate both in gepetto-viewer:
  - world/base_frame
  - ground-truth skeleton (yellow, green lines + spine)
  - estimated skeleton  (red,    red   lines + spine)
"""

import sys
import time
import argparse

import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

# copy your joint names & connections
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

# colors & sizes
GT_JOINT_COLOR   = [1,1,0,1]  # yellow
GT_BONE_COLOR    = [0,1,0,1]  # green
EST_JOINT_COLOR  = [1,0,0,1]  # red
EST_BONE_COLOR   = [1,0,0,1]  # red
BASE_FRAME_COLOR = [255,0,0,1]

JOINT_RAD     = 0.02
BASE_RAD      = 0.02
BASE_LEN      = 0.15
LINE_WIDTH    = 2.0

def gv_init():
    model      = pin.Model()
    geom_model = pin.GeometryModel()
    vis_model  = pin.GeometryModel()
    viz        = GepettoVisualizer(model, geom_model, vis_model)
    try:
        viz.initViewer()
        viz.loadViewerModel("pinocchio")
    except Exception as e:
        print("Error: make sure gepetto-viewer is installed & running.")
        print(e)
        sys.exit(1)
    viz.viewer.gui.addXYZaxis('world/base_frame',
                              BASE_FRAME_COLOR,
                              BASE_RAD, BASE_LEN)
    M0 = pin.SE3(np.eye(3), np.zeros((3,1)))
    viz.viewer.gui.applyConfiguration(
        'world/base_frame',
        pin.SE3ToXYZQUAT(M0).tolist())
    viz.viewer.gui.refresh()
    return viz

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True,
                   help=".npz with 'gt_joints' & 'pred_joints'")
    args = p.parse_args()

    data = np.load(args.npz)
    gt   = data['gt_joints']    # (F,J,3)
    pred = data['pred_joints']  # (F,J,3)
    F, J, _ = gt.shape

    viz = gv_init()

    # create spheres + lines for both
    for j in JOINT_NAMES + ['mid_hip','mid_shoulder']:
        viz.viewer.gui.addSphere(f'world/{j}',
                                 JOINT_RAD, GT_JOINT_COLOR)
        viz.viewer.gui.addSphere(f'world/est_{j}',
                                 JOINT_RAD, EST_JOINT_COLOR)

    def add_lines(prefix, traj, color):
        for a,b in CONNECTIONS:
            name = f'world/{prefix}_line_{a}_{b}'
            p1 = traj[0,JOINT_NAMES.index(a)].tolist()
            p2 = traj[0,JOINT_NAMES.index(b)].tolist()
            viz.viewer.gui.addLine(name, p1, p2, color)
            viz.viewer.gui.setCurveLineWidth(name, LINE_WIDTH)
        # spine
        mh = ((traj[0,JOINT_NAMES.index('r_hip')] +
               traj[0,JOINT_NAMES.index('l_hip')]) / 2).tolist()
        ms = ((traj[0,JOINT_NAMES.index('r_shoulder')] +
               traj[0,JOINT_NAMES.index('l_shoulder')]) / 2).tolist()
        name = f'world/{prefix}_spine'
        viz.viewer.gui.addLine(name, mh, ms, color)
        viz.viewer.gui.setCurveLineWidth(name, LINE_WIDTH)

    add_lines('gt',  gt,  GT_BONE_COLOR)
    add_lines('est', pred, EST_BONE_COLOR)
    viz.viewer.gui.refresh()

    try:
        while True:
            for fidx in range(F):
                # joints
                for j,jn in enumerate(JOINT_NAMES):
                    Mgt  = pin.SE3(np.eye(3),
                                   np.matrix(gt[fidx,j]).T)
                    Mest = pin.SE3(np.eye(3),
                                   np.matrix(pred[fidx,j]).T)
                    viz.viewer.gui.applyConfiguration(
                        f'world/{jn}',
                        pin.SE3ToXYZQUAT(Mgt).tolist())
                    viz.viewer.gui.applyConfiguration(
                        f'world/est_{jn}',
                        pin.SE3ToXYZQUAT(Mest).tolist())
                # midpoints
                mh_gt = ((gt[fidx,JOINT_NAMES.index('r_hip')] +
                          gt[fidx,JOINT_NAMES.index('l_hip')]) / 2).reshape(3,1)
                ms_gt = ((gt[fidx,JOINT_NAMES.index('r_shoulder')] +
                          gt[fidx,JOINT_NAMES.index('l_shoulder')]) / 2).reshape(3,1)
                mh_e  = ((pred[fidx,JOINT_NAMES.index('r_hip')] +
                          pred[fidx,JOINT_NAMES.index('l_hip')]) / 2).reshape(3,1)
                ms_e  = ((pred[fidx,JOINT_NAMES.index('r_shoulder')] +
                          pred[fidx,JOINT_NAMES.index('l_shoulder')]) / 2).reshape(3,1)
                for name,pos in [('mid_hip',mh_gt),('mid_shoulder',ms_gt)]:
                    viz.viewer.gui.applyConfiguration(
                        f'world/{name}',
                        pin.SE3ToXYZQUAT(pin.SE3(np.eye(3), pos)).tolist())
                for name,pos in [('est_mid_hip',mh_e),('est_mid_shoulder',ms_e)]:
                    viz.viewer.gui.applyConfiguration(
                        f'world/{name}',
                        pin.SE3ToXYZQUAT(pin.SE3(np.eye(3), pos)).tolist())
                # lines
                for prefix, traj in [('gt',gt),('est',pred)]:
                    for a,b in CONNECTIONS:
                        p1 = traj[fidx,JOINT_NAMES.index(a)].tolist()
                        p2 = traj[fidx,JOINT_NAMES.index(b)].tolist()
                        viz.viewer.gui.setLineExtremalPoints(
                            f'world/{prefix}_line_{a}_{b}', p1, p2)
                    mh = ((traj[fidx,JOINT_NAMES.index('r_hip')] +
                           traj[fidx,JOINT_NAMES.index('l_hip')]) / 2).tolist()
                    ms = ((traj[fidx,JOINT_NAMES.index('r_shoulder')] +
                           traj[fidx,JOINT_NAMES.index('l_shoulder')]) / 2).tolist()
                    viz.viewer.gui.setLineExtremalPoints(
                        f'world/{prefix}_spine', mh, ms)
                viz.viewer.gui.refresh()
                time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nExiting viewer.")

if __name__ == '__main__':
    main()
