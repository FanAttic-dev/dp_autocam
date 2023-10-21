from enum import Enum
import cv2


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
        1: TEAL,  # player
        2: YELLOW,  # referee
        3: ORANGE,  # goalkeeper
    }


class DrawingMode(Enum):
    LINES = 1
    CIRCLES = 2


INTERPOLATION_TYPE = cv2.INTER_NEAREST
