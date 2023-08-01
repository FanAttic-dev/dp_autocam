from pathlib import Path
import cv2
import numpy as np
import tqdm
from ultralytics import YOLO

from image_processor import ImageProcessor
from constants import colors


class Detector:
    def __init__(self, pitch_coords):
        self.pitch_coords = pitch_coords
        self.bb_ball_last = None

    def preprocess(self, img):
        ...

    def detect(self, frame):
        ...

    def plot(self, bbs):
        ...


class YoloDetector(Detector):
    args = {
        'device': 0,  # 0 if gpu else 'cpu'
        'imgsz': 960,
        'classes': [1, 2, 3, 4],  # [0] for ball only, None for all
        'conf': 0.35,
        'max_det': 50,
        'iou': 0.7
    }
    cls2color = {
        0: colors["red"],  # ball
        1: colors["teal"],  # player
        2: colors["yellow"],  # referee
        3: colors["orange"],  # goalkeeper
    }

    def __init__(self, pitch_coords):
        super().__init__(pitch_coords)
        self.model = YOLO(self.__class__.model_path)

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
                ]
            }
            bbs_frames.append(bbs)
        return bbs_frames

    def draw_bbs_(self, img, bbs):
        for bb, cls in zip(bbs["boxes"], bbs["cls"]):
            x1, y1, x2, y2 = bb
            color = YoloDetector.cls2color[cls]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    def plot(self, res):
        for i in range(len(res)):
            res[i] = res[i].plot()
        return res

    def preprocess(self, img):
        img = ImageProcessor.draw_mask(img, self.pitch_coords, margin=1)
        return img

    def get_ball(self, bbs):
        bb_balls = [
            bb for bb, cls in zip(bbs["boxes"], bbs["cls"]) if cls == 0
        ]

        if len(bb_balls) == 0:
            # no ball detected
            return []

        if not self.bb_ball_last:
            self.bb_ball_last = bb_balls[0]
            return self.bb_ball_last

        # choose the closest detection to the reference
        bb_ball = min(
            bb_balls,
            key=lambda bb: np.linalg.norm(np.array(bb) - np.array(self.bb_ball_last)))

        self.bb_ball_last = bb_ball
        return bb_ball


class YoloBallDetector(YoloDetector):
    args = {
        'device': 0,  # 0 if gpu else 'cpu'
        'imgsz': 960,
        'classes': None,  # [0] for ball only, None for all
        'conf': 0.15,
        'max_det': 1,
        'iou': 0.5
    }

    model_path = Path(
        f"./weights/yolov8_{YoloDetector.args['imgsz']}_ball.pt")

    def detect(self, img):
        res = self.model.predict(
            img, **YoloBallDetector.args, tracker="bytetrack.yaml")
        return self.res2bbs(res), self.plot(res)


class YoloPlayerDetector(YoloDetector):
    model_path = Path(
        f"./weights/yolov8_{YoloDetector.args['imgsz']}.pt")

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

    def __init__(self, pitch_coords):
        super().__init__(pitch_coords)
        self.bgSubtractor = BgDetector.BackgroundSubtractor()

    def preprocess(self, img):
        img = ImageProcessor.draw_mask(img, self.pitch_coords, margin=0)
        img = ImageProcessor.gaussian_blur(img, 1)
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
