from abc import abstractmethod
from utils.config import Config
from detection.image_preprocessor import ImagePreprocessor
import utils.utils as utils


class Detector:
    def __init__(self, frame_orig, top_down, config: Config):
        self.image_preprocessor = ImagePreprocessor(
            frame_orig, top_down, config
        )

        self.detection_blacklist = config.dataset["detection_blacklist"]

    @abstractmethod
    def preprocess(self, img):
        ...

    @abstractmethod
    def detect(self, frame):
        ...

    @abstractmethod
    def plot(self, bbs):
        ...

    @abstractmethod
    def get_stats(self):
        ...

    def filter_detections_(self, bbs):
        i = 0
        while i < len(bbs["boxes"]):
            bb_inner = bbs["boxes"][i]
            for bb_outer in self.detection_blacklist:
                if utils.is_box_in_box(bb_inner, bb_outer):
                    utils.remove_item_in_dict_lists_(bbs, i)
                    i -= 1
                    break
            i += 1
