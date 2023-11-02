import numpy as np
from camera.rectilinear_camera import RectilinearCamera
from utils.constants import Color
from utils.config import Config


class FrameSplitter:
    def __init__(self, frame, config: Config):
        self.cameras = [
            RectilinearCamera(frame, config, ignore_bounds=True).set_ptz(
                pan_deg=camera_params["pan_deg"],
                tilt_deg=camera_params["tilt_deg"],
                zoom_f=camera_params["zoom_f"]
            )
            for camera_params in config.dataset["frame_splitter_params"]
        ]

    def split(self, frame_orig):
        return [camera.get_frame_roi(frame_orig) for camera in self.cameras]

    def flatten_bbs(self, bbs):
        """Take the detections of each camera and join them into one list of detections.

        Args:
            bbs (list): A list of detections for each camera separately.
                        The length must be equal to the number of cameras.

        Returns:
            A dictionary of detections, their classes and track ids.
        """
        assert len(bbs) == len(self.cameras)

        bbs_flattened = {
            "boxes": [],
            "cls": [],
            "ids": []
        }
        for camera, frame_bbs in zip(self.cameras, bbs):
            for i, (bb, cls) in enumerate(zip(frame_bbs["boxes"], frame_bbs["cls"])):
                bb = np.reshape(bb, (2, 2))
                bb_inv = camera.roi2original(bb)
                bb_inv = bb_inv.ravel()

                bbs_flattened["boxes"].append(bb_inv)
                bbs_flattened["cls"].append(cls)

                if len(frame_bbs["ids"]) > 0:
                    id = frame_bbs["ids"][i]
                    bbs_flattened["ids"].append(id)
        return bbs_flattened

    def draw_roi_(self, frame):
        for camera in self.cameras:
            camera.draw_roi_(frame, Color.GREEN)
