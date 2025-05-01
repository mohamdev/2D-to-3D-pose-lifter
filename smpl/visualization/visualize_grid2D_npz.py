#!/usr/bin/env python3
# visualize_grid_npz.py

import os
import sys
import json
import math
import numpy as np
import cv2

# ——— CONFIG ———
CELL_SIZE = 256   # size of each camera view tile (px)

# joint order must match your .npz / your JOINT_NAMES
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

def make_canvas(views_2d, img_w, img_h):
    """
    views_2d: (P, J, 2) array for one frame
    img_w, img_h: original image size used during augmentation
    returns a (rows*CELL_SIZE, cols*CELL_SIZE, 3) uint8 canvas
    """
    P, J, _ = views_2d.shape
    rows = int(math.ceil(math.sqrt(P)))
    cols = int(math.ceil(P / rows))
    canvas_h = rows * CELL_SIZE
    canvas_w = cols * CELL_SIZE
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for p in range(P):
        r = p // cols
        c = p % cols
        x_off = c * CELL_SIZE
        y_off = r * CELL_SIZE

        pts = views_2d[p]  # (J,2)

        # scale from [0,img_w]×[0,img_h] to [0,CELL_SIZE)
        u = (pts[:,0] / img_w) * (CELL_SIZE - 1)
        v = (pts[:,1] / img_h) * (CELL_SIZE - 1)
        uv = np.stack([u, v], axis=1).astype(int)

        # draw bones
        for i,j in SKELETON_EDGES:
            pt1 = (x_off + uv[i,0], y_off + uv[i,1])
            pt2 = (x_off + uv[j,0], y_off + uv[j,1])
            cv2.line(canvas, pt1, pt2, (255,255,255), 1)

        # draw joints
        for (px,py) in uv:
            cv2.circle(canvas, (x_off+px, y_off+py), 1, (0,255,255), -1)

    return canvas

def animate_grid(npz_path):
    # load data
    data = np.load(npz_path)
    joints_2d = data['joints_2d']      # (P, F, J, 2)
    P, F, J, _ = joints_2d.shape

    # read image size from meta if present
    img_w = img_h = None
    if 'meta' in data:
        try:
            meta = json.loads(data['meta'])
            img_w = meta.get('IMG_W', None)
            img_h = meta.get('IMG_H', None)
        except Exception:
            pass
    if img_w is None or img_h is None:
        # fallback to defaults
        img_w, img_h = 1920, 1080

    win = os.path.basename(npz_path)
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    frame_idx = 0

    print(f"Animating '{win}': {F} frames, {P} views per frame")
    print("Hold any key to play, release to pause; press 'q' to quit.")

    while True:
        # extract all P views for this frame
        frame_views = joints_2d[:, frame_idx]  # (P, J, 2)

        canvas = make_canvas(frame_views, img_w, img_h)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # any key → advance
        # if key != 255 and key != 0xFF:
        frame_idx = (frame_idx + 1) % F
        # else no key → stay on same frame

    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python visualize_grid_npz.py <path/to/clip_eM_iL.npz>")
        sys.exit(1)
    animate_grid(sys.argv[1])
