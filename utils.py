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
