from abc import ABC, abstractmethod
from utils.config import Config
from detection.image_preprocessor import ImagePreprocessor


class Detector(ABC):
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
