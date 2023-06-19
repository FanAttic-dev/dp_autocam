import cv2
import tqdm


class Detector:
    def __init__(self):
        ...

    def detect(frame):
        ...


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

    def __init__(self):
        self.bgSubtractor = BgDetector.BackgroundSubtractor()

    def get_bounding_boxes(self, mask):
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        return [cv2.boundingRect(contour) for contour in contours]

    def detect(self, img):
        mask = self.bgSubtractor.apply(img)

        se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                se, iterations=BgDetector.MORPH_CLOSE_ITERATIONS)

        bbs = self.get_bounding_boxes(mask)
        return bbs
