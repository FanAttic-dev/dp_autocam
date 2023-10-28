import math
import numpy as np
import yaml

from utils.constants import DT_FLOAT


def load_yaml(file_name):
    with open(file_name, 'r') as f:
        return yaml.safe_load(f)


def apply_homography(H: np.ndarray, x, y):
    v = np.array([x, y, 1])
    v = H.dot(v)
    x = v[0] / v[2]
    y = v[1] / v[2]
    return x, y


def pts_average(pts, weights=None):
    return np.average(pts, axis=0, weights=weights)


def pts_variance(pts, mu=None, weights=None):
    mu = mu if mu is not None else pts_average(pts, weights)
    return np.average((pts - mu)**2, axis=0, weights=weights)


def discard_extreme_tdpts_(tdpts):
    """Discard extreme top-down points along x-axis.

    Removes one point with the highest and one point with the lowerst x value.
    If the input array contains less than 3 items, no item is removed.
    """
    if len(tdpts["pts"]) < 3:
        return

    maxi = np.argmax(tdpts["pts"], axis=0)[0]
    remove_item_in_dict_lists_(tdpts, maxi)

    mini = np.argmin(tdpts["pts"], axis=0)[0]
    remove_item_in_dict_lists_(tdpts, mini)


def discard_extreme_bbs_(bbs):
    """Discard extreme bounding boxes along x-axis.

    Removes one bb with the highest and one bb with the lowerst x value.
    If the input array contains less than 3 bbs, no bb is removed.
    """
    if len(bbs["boxes"]) < 3:
        return

    maxi = np.argmax(bbs["boxes"], axis=0)[2]
    remove_item_in_dict_lists_(bbs, maxi)

    mini = np.argmin(bbs["boxes"], axis=0)[0]
    remove_item_in_dict_lists_(bbs, mini)


def remove_item_in_dict_lists_(dict: dict, idx) -> None:
    """Remove item in dictionary based on index.

    The dictionary has arrays as values. An item at the given index
    is removed from each array.

    Args:
        dict: A dictionary with lists as values. For example:
            dict = {
                "boxes": [[...], [...], [...]],
                "cls": [0, 3, 2],
                "ids": []
            }
    """
    for k in dict.keys():
        if len(dict[k]) <= idx:
            continue
        dict[k] = np.delete(dict[k], idx, axis=0)


def is_box_in_box(inner, outer) -> bool:
    def check_bounds(u, u_min, u_max):
        return u >= u_min and u <= u_max

    inner_x1, inner_y1, inner_x2, inner_y2 = inner
    outer_x1, outer_y1, outer_x2, outer_y2 = outer

    return check_bounds(inner_x1, outer_x1, outer_x2) and \
        check_bounds(inner_x2, outer_x1, outer_x2) and \
        check_bounds(inner_y1, outer_y1, outer_y2) and \
        check_bounds(inner_y2, outer_y1, outer_y2)


def is_polygon_in_box(inner_poly: np.ndarray, outer_box: np.ndarray):
    def check_bounds(pt: np.ndarray, pt_min: np.ndarray, pt_max: np.ndarray):
        return (pt >= pt_min).all() and (pt <= pt_max).all()

    pt_min, pt_max = outer_box.reshape((2, 2))
    return np.array([check_bounds(pt, pt_min, pt_max) for pt in inner_poly]).all()


def get_bbs_bounding_box(bbs):
    """Returns the bounding box of all input bounding boxes."""
    boxes = np.array(bbs["boxes"])
    x_min, y_min, _, _ = boxes.min(axis=0)
    _, _, x_max, y_max = boxes.max(axis=0)
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
    left_top = pts[1]
    right_top = pts[2]
    u = np.array(right_top - left_top, dtype=DT_FLOAT)
    u /= np.linalg.norm(u)
    v = np.array([[1, 0]])
    return np.arccos(np.dot(u, v.T))


def path2str(path):
    return str(path.absolute())


def mask_out_red_channel(img: np.ndarray):
    img = img.copy()
    img[:, :, 0] = 0
    img[:, :, 1] = 0
    return img


def join_bbs(bbs1, bbs2):
    return {
        "boxes": bbs1["boxes"] + bbs2["boxes"],
        "cls": bbs1["cls"] + bbs2["cls"],
        "ids": bbs1["ids"] + bbs2["ids"]
    }


def hFoV2vFoV(hFoV_deg, aspect_ratio):
    hFoV_rad = np.deg2rad(hFoV_deg)
    vFoV_rad = 2 * np.arctan(np.tan(hFoV_rad / 2) / aspect_ratio)
    return np.rad2deg(vFoV_rad)
