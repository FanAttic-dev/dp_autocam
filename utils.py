import math
import random
import numpy as np
import json


def coords_to_pts(coords):
    pts = np.array([[v["x"], v["y"]] for v in coords.values()], dtype=np.int32)
    return pts.reshape((-1, 1, 2))


def iou(bb1, bb2):
    bb1_x1, bb1_y1, bb1_x2, bb1_y2 = bb1
    bb2_x1, bb2_y1, bb2_x2, bb2_y2 = bb2

    bb1_w = bb1_x2 - bb1_x1 + 1
    bb1_h = bb1_y2 - bb1_y1 + 1

    bb2_w = bb2_x2 - bb2_x1 + 1
    bb2_h = bb2_y2 - bb2_y1 + 1

    bb1_area = bb1_w * bb1_h
    bb2_area = bb2_w * bb2_h

    inner_x1 = max(bb1_x1, bb2_x1)
    inner_y1 = max(bb1_y1, bb2_y1)
    inner_x2 = min(bb1_x2, bb2_x2)
    inner_y2 = min(bb1_y2, bb2_y2)

    intersection = max(0, inner_x2 - inner_x1 + 1) * \
        max(0, inner_y2 - inner_y1 + 1)

    return intersection / float(bb1_area + bb2_area - intersection)


def load_json(file_name):
    with open(file_name, 'r') as f:
        return json.load(f)


def apply_homography(H, x, y):
    v = np.array([x, y, 1])
    v = H.dot(v)
    x = v[0] / v[2]
    y = v[1] / v[2]
    return x, y


def get_random_file(dir):
    files = list(dir.iterdir())
    idx = random.randint(0, len(files)-1)
    return files[idx]


def merge_bbs(bbs1, bbs2):
    return {
        "boxes": bbs1["boxes"] + bbs2["boxes"],
        "cls": bbs1["cls"] + bbs2["cls"]
    }


def add_bb_(bbs, bb, cls):
    if not bb:
        return

    bbs["boxes"].append(bb)
    bbs["cls"].append(cls)


def add_bb_ball_(bbs, bb_ball):
    add_bb_(bbs, bb_ball, 0)


def points_average(points, weights=None):
    return np.average(np.array(points["points"]), axis=0, weights=weights)


def points_variance(points, mu=None, weights=None):
    mu = mu if mu is not None else points_average(points, weights)
    return np.average((points["points"] - mu)**2, axis=0, weights=weights)


def discard_extreme_points_(points):
    pts = points["points"]
    maxi = np.argmax(pts, axis=0)[0]
    mini = np.argmin(pts, axis=0)[0]
    points["points"] = np.delete(points["points"], maxi, axis=0)
    points["points"] = np.delete(points["points"], mini, axis=0)
    points["cls"] = np.delete(points["cls"], maxi, axis=0)
    points["cls"] = np.delete(points["cls"], mini, axis=0)


def lies_in_rectangle(pt, rect):
    start_pt, end_pt = rect
    start_x, start_y = start_pt
    end_x, end_y = end_pt
    pt_x, pt_y = pt
    return start_x <= pt_x and start_y <= pt_y \
        and pt_x <= end_x and pt_y <= end_y


def get_bounding_box(bbs):
    x_min, x_max = np.inf, -np.inf
    y_min, y_max = np.inf, -np.inf

    for bb in bbs["boxes"]:
        x1, y1, x2, y2 = bb
        x_min = min(x_min, x1)
        y_min = min(y_min, y1)
        x_max = max(x_max, x2)
        y_max = max(y_max, y2)

    return x_min, y_min, x_max, y_max


def get_bb_center(bb):
    x1, y1, x2, y2 = bb
    x = (x1 + x2) // 2
    y = (y1 + y2) // 2
    return x, y


def rotate_pts(pts, angle_rad):
    center_x, center_y = np.mean(pts, axis=0)
    pts_rot = []
    for x, y in pts:
        qx = center_x + \
            math.cos(angle_rad) * (x - center_x) - \
            math.sin(angle_rad) * (y - center_y)
        qy = center_y + \
            math.sin(angle_rad) * (x - center_x) + \
            math.cos(angle_rad) * (y - center_y)
        pts_rot.append([qx, qy])
    return pts_rot


def get_pitch_rotation_rad(pitch_coords):
    pts = coords_to_pts(pitch_coords)
    mid_left = pts[1]  # (pts[0] + pts[1]) / 2
    mid_right = pts[4]  # (pts[2] + pts[3]) / 2
    u = np.array(mid_right - mid_left, dtype=np.float64)
    u /= np.linalg.norm(u)
    v = np.array([[1, 0]])
    return np.arccos(np.dot(u, v.T))
