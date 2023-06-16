import numpy as np


def coords_to_pts(coords):
    pts = np.array([[v["x"], v["y"]] for v in coords.values()], np.int32)
    return pts.reshape((-1, 1, 2))
