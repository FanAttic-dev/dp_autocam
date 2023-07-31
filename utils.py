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
