#!/usr/bin/env python3
"""
view_cameras_skeleton.py

Load a .npz with keys 'R','t','joints_3d' and:
 - draw all camera frames (static)
 - animate the 3D skeleton (spheres + green, thick lines) through time
 - use setLineExtremalPoints() to update bones
 - draw a "spine" from mid-hips to mid-shoulders (with its own sphere endpoints)
"""

import sys
import time
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

# joint names must match the original order in joints_3d
JOINT_NAMES = [
    'r_shoulder','l_shoulder',
    'r_elbow',   'l_elbow',
    'r_wrist',   'l_wrist',
    'r_hip',     'l_hip',
    'r_knee',    'l_knee',
    'r_ankle',   'l_ankle',
]

# bones to connect by lines
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

# colors
CAMERA_COLOR    = [0,   0,   255, 1]   # blue
JOINT_COLOR     = [1,   1,   0,   1]   # yellow
BONE_COLOR      = [0,   1,   0,   1]   # green
BASE_FRAME_COLOR= [255, 0,   0,   1]   # red

# line width multiplier (2× default)
LINE_WIDTH = 2.0

def gv_init():
    """Initialize Gepetto visualizer with an empty model and world frame."""
    model      = pin.Model()
    geom_model = pin.GeometryModel()
    vis_model  = pin.GeometryModel()
    viz        = GepettoVisualizer(model, geom_model, vis_model)
    try:
        viz.initViewer()
        viz.loadViewerModel("pinocchio")
    except Exception as e:
        print("Cannot initialize Gepetto-viewer. Is it installed & running?")
        print(e)
        sys.exit(1)

    # world/base_frame
    viz.viewer.gui.addXYZaxis(
        'world/base_frame', BASE_FRAME_COLOR, 0.02, 0.15
    )
    M0 = pin.SE3(np.eye(3), np.zeros((3,1)))
    viz.viewer.gui.applyConfiguration(
        'world/base_frame', pin.SE3ToXYZQUAT(M0).tolist()
    )
    viz.viewer.gui.refresh()
    return viz

def place_frame(viz, name, M):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())

def main(npz_path):
    data       = np.load(npz_path)
    Rs         = data['R']          # (P,3,3)
    ts         = data['t']          # (P,3)
    joints_3d  = data['joints_3d']  # (F,J,3)
    F, J, _    = joints_3d.shape

    if Rs.ndim != 3 or ts.ndim != 2:
        print("Expecting R.shape=(P,3,3), t.shape=(P,3).")
        sys.exit(1)

    print(f"Loaded {Rs.shape[0]} cameras and {F} frames of skeleton.")

    viz = gv_init()

    # 1) draw all camera frames (static)
    for i, (R_w2c, t_w2c) in enumerate(zip(Rs, ts)):
        cam_name = f'world/camera_{i}'
        viz.viewer.gui.addXYZaxis(cam_name, CAMERA_COLOR, 0.03, 0.08)
        R_c2w = R_w2c.T
        C     = - R_c2w @ t_w2c.reshape(3,1)
        M_cam = pin.SE3(R_c2w, C)
        place_frame(viz, cam_name, M_cam)

    # 2) create spheres for each joint + midpoints
    for jn in JOINT_NAMES + ['mid_hip', 'mid_shoulder']:
        viz.viewer.gui.addSphere(f'world/{jn}', 0.02, JOINT_COLOR)

    # 3) create one line per bone + one spine line (initialized at frame 0)
    #    and set their colors & widths
    # 3a) bone lines
    for (a, b) in CONNECTIONS:
        line_name = f'world/line_{a}_{b}'
        p1 = joints_3d[0, JOINT_NAMES.index(a)].tolist()
        p2 = joints_3d[0, JOINT_NAMES.index(b)].tolist()
        viz.viewer.gui.addLine(line_name, p1, p2, BONE_COLOR)
        viz.viewer.gui.setCurveLineWidth(line_name, LINE_WIDTH)

    # 3b) spine line
    mid_hip0       = (joints_3d[0, JOINT_NAMES.index('r_hip')] +
                      joints_3d[0, JOINT_NAMES.index('l_hip')]) / 2.0
    mid_shoulder0  = (joints_3d[0, JOINT_NAMES.index('r_shoulder')] +
                      joints_3d[0, JOINT_NAMES.index('l_shoulder')]) / 2.0
    viz.viewer.gui.addLine('world/line_spine',
                           mid_hip0.tolist(),
                           mid_shoulder0.tolist(),
                           BONE_COLOR)
    viz.viewer.gui.setCurveLineWidth('world/line_spine', LINE_WIDTH)

    viz.viewer.gui.refresh()

    # 4) animation loop
    try:
        while True:
            for f in range(F):
                # 4a) update joint spheres
                for idx, jn in enumerate(JOINT_NAMES):
                    pos = joints_3d[f, idx].reshape(3,1)
                    M   = pin.SE3(np.eye(3), pos)
                    viz.viewer.gui.applyConfiguration(
                        f'world/{jn}', pin.SE3ToXYZQUAT(M).tolist()
                    )
                # 4b) update midpoints
                mid_hip       = ((joints_3d[f, JOINT_NAMES.index('r_hip')] +
                                  joints_3d[f, JOINT_NAMES.index('l_hip')]) / 2.0).reshape(3,1)
                mid_shoulder  = ((joints_3d[f, JOINT_NAMES.index('r_shoulder')] +
                                  joints_3d[f, JOINT_NAMES.index('l_shoulder')]) / 2.0).reshape(3,1)
                for name, pos in [('mid_hip', mid_hip), ('mid_shoulder', mid_shoulder)]:
                    M = pin.SE3(np.eye(3), pos)
                    viz.viewer.gui.applyConfiguration(
                        f'world/{name}', pin.SE3ToXYZQUAT(M).tolist()
                    )

                # 4c) update bone lines
                for (a, b) in CONNECTIONS:
                    p1 = joints_3d[f, JOINT_NAMES.index(a)].tolist()
                    p2 = joints_3d[f, JOINT_NAMES.index(b)].tolist()
                    viz.viewer.gui.setLineExtremalPoints(
                        f'world/line_{a}_{b}', p1, p2
                    )

                # 4d) update spine line
                viz.viewer.gui.setLineExtremalPoints(
                    'world/line_spine',
                    mid_hip.flatten().tolist(),
                    mid_shoulder.flatten().tolist()
                )

                viz.viewer.gui.refresh()
                time.sleep(0.01)   # ~20 FPS

    except KeyboardInterrupt:
        print("\nAnimation stopped. Exiting.")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python view_cameras_skeleton.py path/to/file.npz")
        sys.exit(1)
    main(sys.argv[1])
