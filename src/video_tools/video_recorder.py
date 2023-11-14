from functools import cached_property
from pathlib import Path

import cv2

from camera.camera import Camera
from cameraman.autocam_cameraman import AutocamCameraman
from detection.detector import Detector
from utils.config import Config
from utils.constants import Color
from utils.protocols import HasStats
import utils.utils as utils
from video_tools.video_player import VideoPlayer


class VideoRecorder:
    FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
    IMG_SUFFIX = ".jpg"
    IS_COLOR = True
    STATS_WIDTH = 500
    TOP_DOWN_WIDTH = 250
    TEXT_MARGIN = 20
    TEXT_FORMAT = {
        "fontFace": cv2.FONT_HERSHEY_DUPLEX,
        "fontScale": 0.6,
        "thickness": 1
    }

    def __init__(
        self,
        config: Config,
        video_player: VideoPlayer,
        camera: Camera,
        detector: Detector,
        cameraman: AutocamCameraman
    ):
        self.camera = camera
        self.video_player = video_player
        self.detector = detector
        self.cameraman = cameraman
        self.file_path = config.output_file_path
        self.writer = None
        self.writer_debug = None

    def init_writer(self):
        self.writer = cv2.VideoWriter(
            utils.path2str(self.file_path),
            VideoRecorder.FOURCC,
            self.video_player.fps,
            self.frame_size,
            VideoRecorder.IS_COLOR
        )

        print(f"Video recorder initialized: {utils.path2str(self.file_path)}")

    def init_debug_writer(self):
        self.file_path_debug = self.file_path.with_stem(
            self.file_path.stem + "_debug")
        self.writer_debug = cv2.VideoWriter(
            utils.path2str(self.file_path_debug),
            VideoRecorder.FOURCC,
            self.video_player.fps,
            self.frame_size_debug,
            VideoRecorder.IS_COLOR
        )

        print(
            f"Video debug recorder initialized: {utils.path2str(self.file_path_debug)}")

    @staticmethod
    def get_text_height():
        (_, h), _ = cv2.getTextSize(text="Lorem ipsum", **VideoRecorder.TEXT_FORMAT)
        return h

    @cached_property
    def frame_size(self):
        return self.camera.FRAME_W, self.camera.FRAME_H

    @cached_property
    def frame_size_debug(self):
        return self.camera.FRAME_W + VideoRecorder.STATS_WIDTH, self.camera.FRAME_H

    @cached_property
    def text_x(self):
        frame_w, _ = self.frame_size_debug
        return frame_w - VideoRecorder.STATS_WIDTH + VideoRecorder.TEXT_MARGIN

    @cached_property
    def spacing(self):
        return VideoRecorder.get_text_height() + VideoRecorder.TEXT_MARGIN

    def _add_stats_bar(self, frame_roi):
        def _add_border(frame_roi):
            return cv2.copyMakeBorder(
                frame_roi,
                0, 0, 0, VideoRecorder.STATS_WIDTH,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )

        def _put_dict_items_(frame_roi, dict):
            nonlocal text_y
            for key, value in dict.items():
                text = value if key == "Name" else f"{key}: {value}"
                frame_roi = cv2.putText(
                    img=frame_roi,
                    text=text,
                    org=(self.text_x, text_y),
                    color=Color.WHITE,
                    **VideoRecorder.TEXT_FORMAT
                )
                text_y += self.spacing
            text_y += self.spacing

        def _get_stats(obj: HasStats, name):
            stats = obj.get_stats()
            stats["Name"] = f"{name}: {stats['Name']}"
            return stats

        text_y = self.spacing

        frame_roi = _add_border(frame_roi)

        # _put_dict_items_(frame_roi, _get_stats(self.detector, "Detector"))
        _put_dict_items_(frame_roi, _get_stats(self.camera, "Camera"))
        # put_dict_items_(frame, get_stats(self.camera.pid_x, "PID_X"))
        # put_dict_items_(frame, get_stats(self.camera.pid_y, "PID_Y"))
        _put_dict_items_(frame_roi, _get_stats(self.camera.pid_f, "PID_F"))
        _put_dict_items_(frame_roi, _get_stats(self.cameraman, "Cameraman"))
        _put_dict_items_(frame_roi, _get_stats(
            self.cameraman.ball_filter, "Ball")
        )

        return frame_roi

    def _add_top_down(self, frame_roi, frame_top_down):
        top_down_h, top_down_w, _ = frame_top_down.shape

        top_down_h = int(
            top_down_h * (VideoRecorder.TOP_DOWN_WIDTH / top_down_w))
        top_down_w = VideoRecorder.TOP_DOWN_WIDTH

        frame_top_down_res = cv2.resize(
            frame_top_down, (top_down_w, top_down_h))
        frame_roi = frame_roi.copy()
        frame_roi[0:top_down_h, -top_down_w-1:-1] = frame_top_down_res
        return frame_roi

    def decorate_frame(self, frame_roi, frame_top_down):
        frame_roi = self._add_top_down(frame_roi, frame_top_down)
        frame_roi = self._add_stats_bar(frame_roi)
        return frame_roi

    def write(self, frame):
        assert self.writer is not None
        self.writer.write(frame)

    def write_debug(self, frame):
        assert self.writer_debug is not None
        self.writer_debug.write(frame)

    def save_frame(self, frame, frame_id, suffix=""):
        dir_path = self.file_path.parent / f"{self.file_path.stem}_frames"
        Path.mkdir(dir_path, exist_ok=True)

        filename = f"{self.file_path.stem}_{frame_id:04d}"
        if suffix:
            filename += f"_{suffix}"
        filename += VideoRecorder.IMG_SUFFIX

        filepath = dir_path / filename
        cv2.imwrite(utils.path2str(filepath), frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
        if self.writer_debug is not None:
            self.writer_debug.release()
