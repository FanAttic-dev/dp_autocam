import numpy as np


def coords_to_pts(coords):
    pts = np.array([[v["x"], v["y"]] for v in coords.values()], np.int32)
    return pts.reshape((-1, 1, 2))


def bbs_to_pts(bbs):
    pts = []
    for bb in bbs:
        x, y, w, h = bb
        pts.append([x, y, x+w, y+h])
    pts = np.array(pts, np.float32)
    return pts


def split_frame(frame):
    full_h, full_w, _ = frame.shape
    center_x, center_y = full_w // 2 + 250, full_h // 2 + 180
    h = 450
    w = 1500
    frame = frame[center_y - h // 2: center_y + h // 2, :]
    frames = [
        frame[:, 0:w*2],
        frame[:, center_x - w // 2: center_x + w // 2],
        frame[:, full_w - 1 - w*2: full_w - 1]
    ]
    return frames
