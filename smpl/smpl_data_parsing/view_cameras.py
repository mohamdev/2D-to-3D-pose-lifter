#!/usr/bin/env python3
"""
view_cameras.py

Load a .npz (with 'R' and 't') and display each camera pose
as an XYZ frame in gepetto-viewer.
"""

import sys
import numpy as np
import pinocchio as pin
from pinocchio.visualize import GepettoVisualizer


def gv_init():
    """
    Initialize GepettoVisualizer with an empty model,
    add world frame.
    """
    # empty Pinocchio model + empty geometry models
    model = pin.Model()
    geom_model = pin.GeometryModel()
    visual_model = pin.GeometryModel()

    viz = GepettoVisualizer(model, geom_model, visual_model)
    try:
        viz.initViewer()
        viz.loadViewerModel("pinocchio")
    except Exception as err:
        print("Error initializing Gepetto viewer. Is gepetto-viewer running and installed?")
        print(err)
        sys.exit(1)

    # add world/base_frame
    viz.viewer.gui.addXYZaxis(
        'world/base_frame',
        [255, 0., 0, 1.],  # red
        0.02,              # radius
        0.15               # length
    )
    # place at identity
    M0 = pin.SE3(np.eye(3), np.zeros((3,1)))
    viz.viewer.gui.applyConfiguration(
        'world/base_frame',
        pin.SE3ToXYZQUAT(M0).tolist()
    )
    viz.viewer.gui.refresh()

    return viz


def place_frame(viz, name: str, M: pin.SE3):
    """
    Place the frame at SE3 M under 'world/name'.
    """
    gui = viz.viewer.gui
    gui.applyConfiguration(name, pin.SE3ToXYZQUAT(M).tolist())
    gui.refresh()


def main(npz_path: str):
    # 1) load
    data = np.load(npz_path)
    Rs = data['R']   # (P,3,3)
    ts = data['t']   # (P,3)

    print("ts:", ts)
    if Rs.ndim != 3 or ts.ndim != 2:
        print("Expecting R.shape=(P,3,3), t.shape=(P,3).")
        sys.exit(1)

    P = Rs.shape[0]
    print(f"Loaded {P} camera poses from '{npz_path}'.")

    # 2) init viewer
    viz = gv_init()

    # 3) add & place each camera frame
    for i, (R, t) in enumerate(zip(Rs, ts)):
        frame_name = f'camera_{i}'
        full_name = f'world/{frame_name}'

        # add a colored axis: here blue for cameras
        viz.viewer.gui.addXYZaxis(
            full_name,
            [0., 0., 255., 1.],  # blue
            0.03,                # radius
            0.08                 # length
        )

        R_w2c = R
        t_w2c = t
        R_c2w = R_w2c.T
        cam_pos = - R_c2w @ t_w2c          # 3×1 column vector
        M = pin.SE3(R_c2w, cam_pos)
        # build SE3
        # M = pin.SE3(R, np.matrix(t).T)
        place_frame(viz, full_name, M)

    print("All camera frames added. Close the viewer window or Ctrl+C to exit.")
    # block until user closes
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python view_cameras.py path/to/file.npz")
        sys.exit(1)
    main(sys.argv[1])
