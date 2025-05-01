# visualize_npz.py

import os
import sys
import math
import numpy as np
import cv2

# ——— CONFIG ———
IMG_W, IMG_H = 1920, 1080    # must match your augmentation settings
CELL_SIZE    = 256           # size of each camera view tile

# joints order must match your .npz
JOINT_NAMES = [
    'r_shoulder','l_shoulder',
    'r_elbow',   'l_elbow',
    'r_wrist',   'l_wrist',
    'r_hip',     'l_hip',
    'r_knee',    'l_knee',
    'r_ankle',   'l_ankle',
]

# skeleton connectivity (indices into JOINT_NAMES)
SKELETON_EDGES = [
    (0, 2), (2, 4),   # R shoulder→elbow→wrist
    (1, 3), (3, 5),   # L shoulder→elbow→wrist
    (6, 8), (8,10),   # R hip→knee→ankle
    (7, 9), (9,11),   # L hip→knee→ankle
    (0, 1),           # shoulders
    (6, 7),           # hips
    (0, 6), (1, 7),   # torso sides
]

def make_canvas(joints_2d_frame):
    """
    joints_2d_frame: (N, J, 2) array for one frame
    returns a (rows*CELL_SIZE, cols*CELL_SIZE, 3) uint8 canvas
    """
    N, J, _ = joints_2d_frame.shape
    # compute grid
    rows = int(math.ceil(math.sqrt(N)))
    cols = int(math.ceil(N / rows))
    canvas_h = rows * CELL_SIZE
    canvas_w = cols * CELL_SIZE
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    # draw each view
    for k in range(N):
        r = k // cols
        c = k % cols
        x_off = c * CELL_SIZE
        y_off = r * CELL_SIZE

        pts = joints_2d_frame[k]  # (J,2)
        # scale from [0,IMG] to [0,CELL_SIZE)
        u = (pts[:,0] / IMG_W) * (CELL_SIZE - 1)
        v = (pts[:,1] / IMG_H) * (CELL_SIZE - 1)
        uv = np.stack([u, v], axis=1).astype(int)

        # draw bones
        for i,j in SKELETON_EDGES:
            pt1 = (x_off + uv[i,0], y_off + uv[i,1])
            pt2 = (x_off + uv[j,0], y_off + uv[j,1])
            cv2.line(canvas, pt1, pt2, (255,255,255), 1)

        # draw joints
        for (px,py) in uv:
            cv2.circle(canvas, (x_off+px, y_off+py), 3, (0,255,255), -1)

    return canvas

def animate(npz_path):
    data = np.load(npz_path)
    joints_2d = data['joints_2d']   # (F, N, J, 2)
    F, N, J, _ = joints_2d.shape

    win = os.path.basename(npz_path)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    frame_idx = 0

    print(f"Animating {win}: {F} frames, {N} views per frame.")
    print("→ Hold any key to advance frames, release to pause. Press 'q' to quit.")

    while True:
        canvas = make_canvas(joints_2d[frame_idx])
        cv2.imshow(win, canvas)

        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        # if any key (other than -1) is pressed, advance
        if key != 255 and key != 0xFF:
            frame_idx = (frame_idx + 1) % F
        # else: no key, frame_idx unchanged

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_npz.py <path/to/motion_camXX.npz>")
        sys.exit(1)
    animate(sys.argv[1])
