import numpy as np
from camera import PerspectiveCamera
from constants import colors

from utils import apply_homography, iou


class FrameSplitter:
    def __init__(self):
        ...

    def split(self, frame):
        ...


class PerspectiveFrameSplitter(FrameSplitter):
    def __init__(self, frame):
        self.cameras = [
            PerspectiveCamera(frame, pan_deg=-30, tilt_deg=9),
            PerspectiveCamera(frame, pan_deg=9, tilt_deg=9),
            PerspectiveCamera(frame, pan_deg=49, tilt_deg=9),
        ]

    def split(self, frame):
        frames = [camera.get_frame(frame) for camera in self.cameras]
        return frames

    def join_bbs(self, bbs):
        bbs_joined = {
            "boxes": [],
            "cls": []
        }
        for camera, frame_bbs in zip(self.cameras, bbs):
            for bb, cls in zip(frame_bbs["boxes"], frame_bbs["cls"]):
                H_inv = np.linalg.inv(camera.H)
                x1, y1, x2, y2 = bb
                x1, y1 = apply_homography(H_inv, x1, y1)
                x2, y2 = apply_homography(H_inv, x2, y2)
                bb_inv = [int(x) for x in [x1, y1, x2, y2]]
                bbs_joined["boxes"].append(bb_inv)
                bbs_joined["cls"].append(cls)

        return bbs_joined

    def nms(self, bbs):
        # TODO: remove overlapping bbs
        ...

    def draw_roi_(self, frame):
        for i, camera in enumerate(self.cameras):
            camera.draw_roi_(frame, list(colors.values())[2+i])


class LinearFrameSplitter(FrameSplitter):
    def __init__(self, coords, margin_horiz=200, margin_vert=50, overlap_px=100):
        self.center_x = (coords["left top"]["x"] +
                         coords["right top"]["x"]) // 2
        self.center_y = (coords["left top"]["y"] +
                         coords["left bottom"]["y"]) // 2

        self.h = coords["left bottom"]["y"] - \
            coords["left top"]["y"] + margin_vert
        self.w = coords["right top"]["x"] - \
            coords["left top"]["x"] + margin_horiz

        self.top = self.center_y - self.h // 2
        self.bottom = self.center_y + self.h // 2

        self.center_start = self.center_x - self.w // 2
        self.center_end = self.center_x + self.w // 2

        self.left_start = coords["left bottom"]["x"]
        self.left_end = self.center_start + overlap_px

        self.right_start = self.center_end - overlap_px
        self.right_end = coords["right bottom"]["x"]

    def crop_sky(self, frame):
        return frame[self.top: self.bottom, :]

    def split(self, frame):
        self.frame_shape = frame.shape
        frame = self.crop_sky(frame)
        frames = [
            frame[:, self.left_start: self.left_end],
            frame[:, self.center_start: self.center_end],
            frame[:, self.right_start: self.right_end]
        ]
        return frames

    def join(self, frames):
        res = np.zeros(self.frame_shape, np.uint8)
        left, center, right = frames
        res[self.top: self.bottom, self.left_start:self.left_end] = left
        res[self.top: self.bottom, self.center_start:self.center_end] = center
        res[self.top: self.bottom, self.right_start:self.right_end] = right
        return res

    def is_in_overlap(self, bb):
        x1, y1, x2, y2 = bb
        x = (x1 + x2) // 2
        return (x <= self.left_end and x >= self.center_start) or (x >= self.right_start and x <= self.center_end)

    def nms(self, bbs, t=0.01):
        # res = [bb1 for bb2 in bbs if iou(bb1, bb2) > t for bb1 in bbs]
        # return res

        res = []
        for bb1 in bbs:
            for bb2 in bbs:
                if bb1 != bb2 and iou(bb1, bb2) > t and bb1 not in res and bb2 not in res:
                    res.append(bb2)
                    break

        return res

    def join_bbs(self, frame_bbs):
        bbs_overlapping = []
        res = []
        bbs_left, bbs_center, bbs_right = frame_bbs
        for i in range(len(bbs_left)):
            x1, y1, x2, y2 = bbs_left[i]
            bb = [
                x1 + self.left_start,
                y1 + self.top,
                x2 + self.left_start,
                y2 + self.top
            ]
            if self.is_in_overlap(bb):
                bbs_overlapping.append(bb)
            else:
                res.append(bb)

        for i in range(len(bbs_center)):
            x1, y1, x2, y2 = bbs_center[i]
            bb = [
                x1 + self.center_start,
                y1 + self.top,
                x2 + self.center_start,
                y2 + self.top
            ]
            if self.is_in_overlap(bb):
                bbs_overlapping.append(bb)
            else:
                res.append(bb)

        for i in range(len(bbs_right)):
            x1, y1, x2, y2 = bbs_right[i]
            bb = [
                x1 + self.right_start,
                y1 + self.top,
                x2 + self.right_start,
                y2 + self.top
            ]
            if self.is_in_overlap(bb):
                bbs_overlapping.append(bb)
            else:
                res.append(bb)

        res += self.nms(bbs_overlapping)
        return res
