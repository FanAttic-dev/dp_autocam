from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

import utils.utils as utils
from utils.constants import Color
from utils.config import Config
from detection.detector import Detector


class YoloDetector(Detector):
    args = {
        'device': torch.cuda.current_device(),
        'imgsz': 960,
        'classes': [1, 2, 3] if Config.autocam["detector"]["ball"]["enabled"] else None,
        'conf': Config.autocam["detector"]["conf"],
        'max_det': 50,
        'iou': 0.5,
        'verbose': False
    }

    model_path = Path(f"./assets/weights/yolov8_{args['imgsz']}.pt")

    def __init__(self, frame_orig, top_down, config):
        super().__init__(frame_orig, top_down, config)
        self.model = YOLO(self.__class__.model_path)

    def preprocess(self, img):
        img = self.image_preprocessor.draw_mask(img)
        return img

    def _res2bbs(self, res):
        bbs_frames = []
        for det_frame in res:
            bbs = {
                "boxes": [
                    bb.cpu().numpy().astype(int)
                    for bb in det_frame.boxes.xyxy
                ],
                "cls": [
                    cls.cpu().numpy().astype(int).item()
                    for cls in det_frame.boxes.cls
                ],
                "ids": [
                    id.cpu().numpy().astype(int).item()
                    for id in det_frame.boxes.id
                ] if det_frame.boxes.is_track else []
            }
            bbs_frames.append(bbs)
        return bbs_frames

    def draw_bbs_(self, img, bbs, color=None):
        if bbs is None:
            return

        for i, (bb, cls) in enumerate(zip(bbs["boxes"], bbs["cls"])):
            x1, y1, x2, y2 = bb
            bb_color = Color.cls2color[cls] if color is None else color
            cv2.rectangle(img, (x1, y1), (x2, y2), bb_color, 2)

            if i >= len(bbs["ids"]):
                continue

            id = bbs["ids"][i]
            cv2.putText(
                img=img,
                text=f"{id}",
                org=(x1, y1),
                color=Color.WHITE,
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.3,
                thickness=5
            )

    def detect(self, img):
        res = self.model.predict(img, **self.__class__.args)
        return self._res2bbs(res), self.plot(res)

    def plot(self, res):
        for i in range(len(res)):
            res[i] = res[i].plot()
        return res

    def filter_detections_(self, bbs):
        """Discard detections that have their bounding boxes inside a blacklisted bounding box."""
        i = 0
        while i < len(bbs["boxes"]):
            bb_inner = bbs["boxes"][i]
            for bb_outer in self.detection_blacklist:
                if utils.is_box_in_box(bb_inner, bb_outer):
                    utils.remove_item_in_dict_lists_(bbs, i)
                    i -= 1
                    break
            i += 1

    def get_stats(self):
        return {
            "Name": self.__class__.__name__,
            "imgsz": self.args["imgsz"],
            "conf": self.args["conf"],
            "max_det": self.args["max_det"],
            "iou": self.args["iou"]
        }


class YoloBallDetector(YoloDetector):
    args = {
        'device': torch.cuda.current_device(),
        'imgsz': 960,
        'classes': None,  # [0] for ball only, None for all
        'conf': Config.autocam["detector"]["ball"]["conf"],
        'max_det': Config.autocam["detector"]["ball"]["max_det"],
        'iou': 0.5,
        'verbose': False
    }

    model_path = Path(
        f"./assets/weights/yolov8_{args['imgsz']}_ball.pt")
