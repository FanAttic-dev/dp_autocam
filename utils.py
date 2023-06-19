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


def split_frame(frame, coords, margin_horiz=-1300, margin_vert=20):
    full_h, full_w, _ = frame.shape
    anchor_left = (coords["left top"]["x"] + coords["left bottom"]["x"]) // 2
    anchor_right = (coords["right top"]["x"] +
                    coords["right bottom"]["x"]) // 2
    # center_x, center_y = full_w // 2 + 250, full_h // 2 + 180
    center_x = (anchor_left + anchor_right) // 2
    center_y = (coords["left top"]["y"] + coords["left bottom"]["y"]) // 2
    # h = 450
    h = coords["left bottom"]["y"] - coords["left top"]["y"] + margin_vert
    # w = 1500
    w = anchor_right - anchor_left + margin_horiz
    print(w)
    frame = frame[center_y - h // 2: center_y + h // 2, :]
    frames = [
        frame[:, 0:w],
        frame[:, center_x - w // 2: center_x + w // 2],
        frame[:, full_w - 1 - w: full_w - 1]
    ]
    return frames
