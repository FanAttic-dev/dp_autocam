from enum import Enum
import cv2
import numpy as np


class Color:
    @staticmethod
    def hex2bgr(hex):
        hex = hex.lstrip("#")
        r, g, b = tuple(int(hex[i:i+2], 16) for i in (0, 2, 4))
        return b, g, r

    WHITE = hex2bgr("#ffffff")
    RED = hex2bgr("#e74645")
    ORANGE = hex2bgr("#fb7756")
    YELLOW = hex2bgr("#fdfa66")
    TEAL = hex2bgr("#1ac0c6")
    VIOLET = hex2bgr("#e645be")
    BLUE = hex2bgr("#456de6")
    GREEN = hex2bgr("#45e645")

    cls2color = {
        0: WHITE,  # ball
        1: YELLOW,  # player
        2: ORANGE,  # referee
        3: TEAL,  # goalkeeper
    }


class DrawingMode(Enum):
    LINES = 1
    CIRCLES = 2


INTERPOLATION_TYPE = cv2.INTER_NEAREST
DT_FLOAT = np.float32
DT_INT = np.int32
VIDEO_SUFFIX = ".mp4"
