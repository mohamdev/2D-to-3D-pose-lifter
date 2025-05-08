#!/usr/bin/env python3
"""
view_cameras_skeleton.py

Load a .npz with keys 'joints_3d', then:
 - randomly generate N camera extrinsics reproducibly
 - draw all camera frames (static)
 - animate the 3D skeleton (spheres + green, thick lines) through time
 - update bones with setLineExtremalPoints()
 - draw a "spine" from mid-hips to mid-shoulders
"""

import sys
import time
import argparse
import numpy as np
import random
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer

# joint names must match original order in joints_3d
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
CAMERA_COLOR     = [0,   0,   255, 1]   # blue
JOINT_COLOR      = [1,   1,   0,   1]   # yellow
BONE_COLOR       = [0,   1,   0,   1]   # green
BASE_FRAME_COLOR = [255, 0,   0,   1]   # red
LINE_WIDTH       = 2.0  # 2× default


def fibonacci_dirs(n: int) -> np.ndarray:
    """
    Generate nearly-uniform directions on the unit sphere via a Fibonacci lattice.
    Returns array of shape (n,3).
    """
    i = np.arange(n)
    phi = (1 + 5**0.5) / 2
    theta = 2 * np.pi * i / phi
    z = 1 - 2*(i + 0.5)/n
    r = np.sqrt(1 - z*z)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)


def look_at_R(cam_pos: np.ndarray, target: np.ndarray = np.zeros(3), up: np.ndarray = np.array([0,1,0])) -> np.ndarray:
    """
    Compute world->camera rotation so the camera at cam_pos looks at target.
    """
    z = target - cam_pos
    z /= np.linalg.norm(z) + 1e-8
    x = np.cross(up, z)
    x /= np.linalg.norm(x) + 1e-8
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=0)


def gv_init():
    """Initialize Gepetto visualizer and draw world frame."""
    model = pin.Model()
    gmodel = pin.GeometryModel()
    vmodel = pin.GeometryModel()
    viz = GepettoVisualizer(model, gmodel, vmodel)
    try:
        viz.initViewer()
        viz.loadViewerModel("pinocchio")
    except Exception as e:
        print("Cannot initialize Gepetto-gui. Is it running?", e)
        sys.exit(1)

    viz.viewer.gui.addXYZaxis('world/base_frame', BASE_FRAME_COLOR, 0.02, 0.15)
    M0 = pin.SE3(np.eye(3), np.zeros((3,1)))
    viz.viewer.gui.applyConfiguration('world/base_frame', pin.SE3ToXYZQUAT(M0).tolist())
    viz.viewer.gui.refresh()
    return viz


def place_frame(viz: GepettoVisualizer, name: str, M: pin.SE3):
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npz_path', help='Path to .npz with joints_3d')
    parser.add_argument('-n', '--num_cams', type=int, default=4,
                        help='Number of random cameras to generate')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (int); if unset, uses a random one')
    parser.add_argument('--dist-min', type=float, default=1.5,
                        help='Min distance factor relative to body radius')
    parser.add_argument('--dist-max', type=float, default=3.5,
                        help='Max distance factor relative to body radius')
    args = parser.parse_args()

    # Seed RNGs
    seed = args.seed if args.seed is not None else random.SystemRandom().randint(0,2**31)
    print(f"Using seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)

    # Load joints_3d
    data = np.load(args.npz_path)
    if 'joints_3d' not in data:
        print("Error: 'joints_3d' key not in npz.")
        sys.exit(1)
    joints_3d = data['joints_3d']  # (F,J,3)
    F, J, _ = joints_3d.shape

    # Compute body radius for camera placement
    R_body = np.linalg.norm(joints_3d.reshape(-1,3), axis=1).max()
    cam_dirs = fibonacci_dirs(args.num_cams)

    # Build extrinsics with varying distance
    extrinsics = []
    for v in cam_dirs:
        factor = random.uniform(args.dist_min, args.dist_max)
        cam_pos = v * (R_body * factor)
        R_w2c = look_at_R(cam_pos)
        t_w2c = -R_w2c.dot(cam_pos)
        extrinsics.append((R_w2c, t_w2c))

    print(f"Generated {len(extrinsics)} camera extrinsics with distance factor in [{args.dist_min}, {args.dist_max}].")
    viz = gv_init()

    # Draw camera frames
    for i, (R_w2c, t_w2c) in enumerate(extrinsics):
        name = f'world/camera_{i}'
        viz.viewer.gui.addXYZaxis(name, CAMERA_COLOR, 0.03, 0.08)
        R_c2w = R_w2c.T
        C = -R_c2w @ t_w2c.reshape(3,1)
        M_cam = pin.SE3(R_c2w, C)
        place_frame(viz, name, M_cam)

    # Add joint spheres
    for jn in JOINT_NAMES + ['mid_hip','mid_shoulder']:
        viz.viewer.gui.addSphere(f'world/{jn}', 0.02, JOINT_COLOR)

    # Initialize bones and spine lines at frame 0
    for (a,b) in CONNECTIONS:
        p1 = joints_3d[0, JOINT_NAMES.index(a)].tolist()
        p2 = joints_3d[0, JOINT_NAMES.index(b)].tolist()
        lname = f'world/line_{a}_{b}'
        viz.viewer.gui.addLine(lname, p1, p2, BONE_COLOR)
        viz.viewer.gui.setCurveLineWidth(lname, LINE_WIDTH)
    mid_hip0 = ((joints_3d[0, JOINT_NAMES.index('r_hip')] +
                 joints_3d[0, JOINT_NAMES.index('l_hip')]) / 2.0).tolist()
    mid_sh0 = ((joints_3d[0, JOINT_NAMES.index('r_shoulder')] +
                joints_3d[0, JOINT_NAMES.index('l_shoulder')]) / 2.0).tolist()
    viz.viewer.gui.addLine('world/line_spine', mid_hip0, mid_sh0, BONE_COLOR)
    viz.viewer.gui.setCurveLineWidth('world/line_spine', LINE_WIDTH)
    viz.viewer.gui.refresh()

    # Animate skeleton
    try:
        while True:
            for f in range(F):
                for idx, jn in enumerate(JOINT_NAMES):
                    pos = joints_3d[f, idx].reshape(3,1)
                    M = pin.SE3(np.eye(3), pos)
                    viz.viewer.gui.applyConfiguration(f'world/{jn}', pin.SE3ToXYZQUAT(M).tolist())
                mid_hip = ((joints_3d[f, JOINT_NAMES.index('r_hip')] +
                            joints_3d[f, JOINT_NAMES.index('l_hip')]) / 2.0).reshape(3,1)
                mid_sh = ((joints_3d[f, JOINT_NAMES.index('r_shoulder')] +
                           joints_3d[f, JOINT_NAMES.index('l_shoulder')]) / 2.0).reshape(3,1)
                for name, pos in [('mid_hip',mid_hip),('mid_shoulder',mid_sh)]:
                    M = pin.SE3(np.eye(3), pos)
                    viz.viewer.gui.applyConfiguration(f'world/{name}', pin.SE3ToXYZQUAT(M).tolist())
                for (a,b) in CONNECTIONS:
                    p1 = joints_3d[f, JOINT_NAMES.index(a)].tolist()
                    p2 = joints_3d[f, JOINT_NAMES.index(b)].tolist()
                    viz.viewer.gui.setLineExtremalPoints(f'world/line_{a}_{b}', p1, p2)
                viz.viewer.gui.setLineExtremalPoints('world/line_spine',
                    mid_hip.flatten().tolist(), mid_sh.flatten().tolist())
                viz.viewer.gui.refresh()
                time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nAnimation stopped.")


if __name__ == '__main__':
    main()
