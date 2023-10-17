import math
import numpy as np
import yaml


def coords2pts(coords):
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


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        return yaml.safe_load(f)


def apply_homography(H, x, y):
    v = np.array([x, y, 1])
    v = H.dot(v)
    x = v[0] / v[2]
    y = v[1] / v[2]
    return x, y


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
    if len(points["points"]) < 3:
        return

    maxi = np.argmax(points["points"], axis=0)[0]
    remove_item_in_dict_lists_(points, maxi)

    mini = np.argmin(points["points"], axis=0)[0]
    remove_item_in_dict_lists_(points, mini)


def discard_extreme_boxes_(bbs):
    maxi = np.argmax(bbs["boxes"], axis=0)[2]
    remove_item_in_dict_lists_(bbs, maxi)

    mini = np.argmin(bbs["boxes"], axis=0)[0]
    remove_item_in_dict_lists_(bbs, mini)


def remove_item_in_dict_lists_(dict, idx):
    for k in dict.keys():
        if len(dict[k]) <= idx:
            continue
        dict[k] = np.delete(dict[k], idx, axis=0)


def boxes_overlap(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1
    b2_x1, b2_y1, b2_x2, b2_y2 = b2

    if b1_x1 > b2_x2 or b2_x1 > b1_x2:
        return False

    if b1_y2 > b2_y1 or b2_y2 > b1_y1:
        return False

    return True


def pt_lies_in_box(pt, box):
    start_pt, end_pt = box
    start_x, start_y = start_pt
    end_x, end_y = end_pt
    pt_x, pt_y = pt
    return start_x <= pt_x and start_y <= pt_y \
        and pt_x <= end_x and pt_y <= end_y


def box_lies_in_box(inner, outer):
    def check_bounds(u, u_min, u_max):
        return u >= u_min and u <= u_max
    inner_x1, inner_y1, inner_x2, inner_y2 = inner
    outer_x1, outer_y1, outer_x2, outer_y2 = outer

    return check_bounds(inner_x1, outer_x1, outer_x2) and \
        check_bounds(inner_x2, outer_x1, outer_x2) and \
        check_bounds(inner_y1, outer_y1, outer_y2) and \
        check_bounds(inner_y2, outer_y1, outer_y2)


def polygon_lies_in_box(inner_poly, outer_box):
    def check_bounds(pt: np.ndarray, pt_min: np.ndarray, pt_max: np.ndarray):
        return (pt >= pt_min).all() and (pt <= pt_max).all()

    pt_min, pt_max = outer_box.reshape((2, 2))
    return np.array([check_bounds(pt, pt_min, pt_max) for pt in inner_poly]).all()


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


def rotate_pts(pts, angle_rad, center=None):
    if center is None:
        center = np.mean(pts, axis=0)

    center_x, center_y = center
    angle_cos = math.cos(angle_rad)
    angle_sin = math.sin(angle_rad)

    pts[:, 0] = center_x + \
        angle_cos * (pts[:, 0] - center_x) - \
        angle_sin * (pts[:, 1] - center_y)
    pts[:, 1] = center_y + \
        angle_sin * (pts[:, 0] - center_x) + \
        angle_cos * (pts[:, 1] - center_y)

    return pts


def get_pitch_rotation_rad(pts):
    if pts is dict:
        pts = coords2pts(pts)
    left_top = pts[1]
    right_top = pts[2]
    u = np.array(right_top - left_top, dtype=np.float64)
    u /= np.linalg.norm(u)
    v = np.array([[1, 0]])
    return np.arccos(np.dot(u, v.T))


def get_bbs_ball(bbs_joined):
    bbs_ball = {
        "boxes": [],
        "cls": [],
        "ids": []
    }
    bbs_joined_new = {
        "boxes": [],
        "cls": [],
        "ids": []
    }
    for bb, cls in zip(bbs_joined["boxes"], bbs_joined["cls"]):
        target = bbs_ball if cls == 0 else bbs_joined_new
        target["boxes"].append(bb)
        target["cls"].append(cls)
    return bbs_ball, bbs_joined_new


def path2str(path):
    return str(path.absolute())
