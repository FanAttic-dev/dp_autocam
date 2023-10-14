import numpy as np
from camera.cyllindrical_camera import CyllindricalCamera
from camera.spherical_camera import SphericalCamera
from utils.constants import Color
from utils.config import Config

from utils.helpers import apply_homography, iou


class FrameSplitter:
    def __init__(self, frame, config: Config):
        self.cameras = [
            SphericalCamera(frame, config).set_ptz(
                pan_deg=camera_params["pan_deg"],
                tilt_deg=camera_params["tilt_deg"],
                zoom_f=camera_params["zoom_f"]
            )
            for camera_params in config.dataset["frame_splitter_params"]
        ]

    def split(self, frame):
        frames = [camera.get_frame(frame) for camera in self.cameras]
        return frames

    def join_bbs(self, bbs):
        bbs_joined = {
            "boxes": [],
            "cls": [],
            "ids": []
        }
        for camera, frame_bbs in zip(self.cameras, bbs):
            for i, (bb, cls) in enumerate(zip(frame_bbs["boxes"], frame_bbs["cls"])):
                bb = bb.reshape((2, 2))
                bb_inv = camera.roi2original(bb)
                bb_inv = bb_inv.ravel()
                bbs_joined["boxes"].append(bb_inv)
                bbs_joined["cls"].append(cls)
                if len(frame_bbs["ids"]) > 0:
                    id = frame_bbs["ids"][i]
                    bbs_joined["ids"].append(id)

        return bbs_joined

    def nms(self, bbs):
        # TODO: remove overlapping bbs
        ...

    def draw_roi_(self, frame):
        for i, camera in enumerate(self.cameras):
            camera.draw_roi_(frame, Color.GREEN)
            # camera.draw_roi_(frame, list(colors.values())[2+i])
