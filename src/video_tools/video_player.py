from functools import cached_property
from pathlib import Path
import cv2


class VideoPlayer:
    WINDOW_NAME = 'frame'
    WINDOW_FLAGS = cv2.WINDOW_NORMAL  # cv2.WINDOW_AUTOSIZE

    def __init__(self, video_path: Path):
        self.video_path = video_path
        self.video_path_str = str(video_path.absolute())
        self.cap = cv2.VideoCapture(self.video_path_str)
        self._mouse_pos = {
            "x": 0,
            "y": 0
        }
        print(f"Video player initialized: {self.video_path.absolute()}")

    @property
    def mouse_pos(self):
        return self._mouse_pos["x"], self._mouse_pos["y"]

    @cached_property
    def fps(self):
        return self.cap.get(cv2.CAP_PROP_FPS)

    def restart(self):
        return self.cap.open(self.video_path_str)

    @property
    def frame_size(self):
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        return int(width), int(height)

    def get_frame_at(self, seconds):
        frame_id = int(self.fps*seconds)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        return self.get_next_frame(self.cap)

    def get_next_frame(self):
        ret, frame = self.cap.read()
        return ret, frame

    def play_video(self):
        while True:
            ret, frame = self.get_next_frame(self.cap)
            if not ret:
                break
            cv2.imshow('frame', frame)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

    def create_window(self, window_name):
        cv2.namedWindow(window_name, VideoPlayer.WINDOW_FLAGS)

    def init_mouse(self, window_name):
        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self._mouse_pos["x"] = x
                self._mouse_pos["y"] = y

        self.create_window(window_name)
        cv2.setMouseCallback("Original", mouse_callback)

    def show_frame(self, frame, window_name=WINDOW_NAME):
        self.create_window(window_name)
        cv2.imshow(window_name, frame)

    def get_delay(self, is_recording=False):
        if is_recording:
            return int(1000 / self.fps)
        return 0

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
