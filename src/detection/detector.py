from pathlib import Path
import cv2
import numpy as np
import tqdm
import torch
from ultralytics import YOLO
from utils.config import Config

from detection.image_preprocessor import ImagePreprocessor
from utils.constants import colors
from utils.helpers import get_bb_center


class Detector:
    def __init__(self, frame_orig, top_down, config):
        self.image_preprocessor = ImagePreprocessor(
            frame_orig, top_down, config)

    def preprocess(self, img):
        ...

    def detect(self, frame):
        ...

    def plot(self, bbs):
        ...

    def get_stats(self):
        ...


class YoloDetector(Detector):
    args = {
        'device': torch.cuda.current_device(),  # 0 if gpu else 'cpu'
        'imgsz': 960,
        'classes': [0, 1, 2, 3],  # [0] for ball only, None for all
        'conf': Config.autocam["detector"]["players_confidence"],
        'max_det': 50,
        'iou': 0.5,
        'verbose': False
    }
    cls2color = {
        0: colors["white"],  # ball
        1: colors["teal"],  # player
        2: colors["yellow"],  # referee
        3: colors["orange"],  # goalkeeper
    }

    def __init__(self, frame_orig, top_down, config):
        super().__init__(frame_orig, top_down, config)
        self.model = YOLO(self.__class__.model_path)

    def preprocess(self, img):
        img = self.image_preprocessor.draw_mask(img)
        return img

    def res2bbs(self, res):
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
        for i, (bb, cls) in enumerate(zip(bbs["boxes"], bbs["cls"])):
            x1, y1, x2, y2 = bb
            bb_color = YoloDetector.cls2color[cls] if color is None else color
            cv2.rectangle(img, (x1, y1), (x2, y2), bb_color, 2)

            if i >= len(bbs["ids"]):
                continue

            id = bbs["ids"][i]
            cv2.putText(
                img=img,
                text=f"{id}",
                org=(x1, y1),
                color=colors["white"],
                fontFace=cv2.FONT_HERSHEY_DUPLEX,
                fontScale=0.3,
                thickness=1
            )

    def plot(self, res):
        for i in range(len(res)):
            res[i] = res[i].plot()
        return res

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
        'device': torch.cuda.current_device(),  # 0 if gpu else 'cpu'
        'imgsz': 960,
        'classes': None,  # [0] for ball only, None for all
        'conf': Config.autocam["detector"]["ball_confidence"],
        'max_det': Config.autocam["detector"]["ball_max_det"],
        'iou': 0.5
    }

    model_path = Path(
        f"./assets/weights/yolov8_{YoloDetector.args['imgsz']}_ball.pt")

    def __init__(self, frame_orig, top_down, config, ball_filter):
        super().__init__(frame_orig, top_down, config)
        self.ball_filter = ball_filter
        self.__ball_threshold = 5

    @property
    def ball_threshold(self):
        th_max = 1000
        th_min = 5
        th = th_max * self.ball_filter.K_x

        dt = 20
        if self.__ball_threshold < th:
            self.__ball_threshold += dt
        else:
            self.__ball_threshold -= dt

        return np.clip(self.__ball_threshold, th_min, th_max)

    def detect(self, img):
        res = self.model.predict(
            img, **YoloBallDetector.args, tracker="bytetrack.yaml")
        return self.res2bbs(res), self.plot(res)

    def draw_ball_radius_(self, frame, color):
        x, y = self.ball_filter.pos
        cv2.circle(frame, (int(x), int(y)), int(self.ball_threshold), color)


class YoloPlayerDetector(YoloDetector):
    model_path = Path(
        f"./assets/weights/yolov8_{YoloDetector.args['imgsz']}.pt")

    def detect(self, img):
        res = self.model.predict(img, **YoloPlayerDetector.args)
        return self.res2bbs(res), self.plot(res)


class BgDetector(Detector):
    MORPH_CLOSE_ITERATIONS = 10

    class BackgroundSubtractor:
        def __init__(self):
            self.model = cv2.createBackgroundSubtractorKNN(history=1)

        def init(self, cap, iterations=10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            for _ in tqdm(range(iterations)):
                ret, frame = cap.read()
                if not ret:
                    return
                self.apply(frame)

                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        def apply(self, img):
            return self.model.apply(img)

    def __init__(self, frame_orig, top_down, config):
        super().__init__(frame_orig, top_down, config)
        self.bgSubtractor = BgDetector.BackgroundSubtractor()

    def preprocess(self, img):
        img = self.image_preprocessor.draw_mask(img)
        img = cv2.GaussianBlur(img, (3, 3), 1)
        return img

    def draw_bounding_boxes(self, img, bbs):
        for bb in bbs:
            x, y, w, h = bb
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 255), 2)

    def get_bounding_boxes(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        bbs = [cv2.boundingRect(contour) for contour in contours]
        return self.bbs2pts(bbs)

    def bbs2pts(self, bbs):
        pts = []
        for bb in bbs:
            x, y, w, h = bb
            pts.append([x, y, x+w, y+h])
        pts = np.array(pts, np.float32)
        return pts

    def detect(self, img):
        img = self.preprocess(img)
        mask = self.bgSubtractor.apply(img)

        se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            se,
            iterations=BgDetector.MORPH_CLOSE_ITERATIONS
        )

        bbs = self.get_bounding_boxes(mask)
        plotted = img.copy()
        self.draw_bounding_boxes(plotted, bbs)
        return bbs, plotted