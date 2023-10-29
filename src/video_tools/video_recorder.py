from functools import cached_property
from pathlib import Path
import cv2
from algorithm.autocam_algo import AutocamAlgo
from camera.camera import Camera
from detection.detector import Detector
from utils.argparse import AutocamArgsNamespace
from utils.config import Config
from utils.constants import Color
from utils.protocols import HasStats
import utils.utils as utils
from video_tools.video_player import VideoPlayer


class VideoRecorder:
    FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
    VIDEO_SUFFIX = ".mp4"
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
        algo: AutocamAlgo
    ):
        self.camera = camera
        self.video_player = video_player
        self.detector = detector
        self.algo = algo
        self.file_path = self.get_file_path(video_player, config.output_dir)
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

    def get_file_path(self, video_player: VideoPlayer, output_dir: str) -> Path:
        output_dir.mkdir(exist_ok=True, parents=True)

        video_path = video_player.video_path.stem
        file_path = output_dir / video_path
        file_path = file_path.with_suffix(VideoRecorder.VIDEO_SUFFIX)

        idx = 1
        while file_path.exists():
            file_path = file_path.with_stem(f"{video_path}_{idx:02}")
            idx += 1

        return file_path

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

    def add_stats_bar(self, frame):
        def add_border(frame):
            return cv2.copyMakeBorder(
                frame,
                0, 0, 0, VideoRecorder.STATS_WIDTH,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )

        def put_dict_items_(frame, dict):
            nonlocal text_y
            for key, value in dict.items():
                text = value if key == "Name" else f"{key}: {value}"
                frame = cv2.putText(
                    img=frame,
                    text=text,
                    org=(self.text_x, text_y),
                    color=Color.WHITE,
                    **VideoRecorder.TEXT_FORMAT
                )
                text_y += self.spacing
            text_y += self.spacing

        def get_stats(obj: HasStats, name):
            stats = obj.get_stats()
            stats["Name"] = f"{name}: {stats['Name']}"
            return stats

        text_y = self.spacing

        frame = add_border(frame)

        # put_dict_items_(frame, get_stats(self.detector, "Detector"))
        put_dict_items_(frame, get_stats(self.camera, "Camera"))
        # put_dict_items_(frame, get_stats(self.camera.pid_x, "PID_X"))
        # put_dict_items_(frame, get_stats(self.camera.pid_y, "PID_Y"))
        put_dict_items_(frame, get_stats(self.camera.pid_f, "PID_F"))
        put_dict_items_(frame, get_stats(self.algo, "Algo"))
        put_dict_items_(frame, get_stats(self.algo.ball_filter, "Ball"))

        return frame

    def add_top_down(self, frame, top_down_frame):
        top_down_h, top_down_w, _ = top_down_frame.shape

        top_down_h = int(
            top_down_h * (VideoRecorder.TOP_DOWN_WIDTH / top_down_w))
        top_down_w = VideoRecorder.TOP_DOWN_WIDTH

        top_down_frame_res = cv2.resize(
            top_down_frame, (top_down_w, top_down_h))
        frame = frame.copy()
        frame[0:top_down_h, -top_down_w-1:-1] = top_down_frame_res
        return frame

    def decorate_frame(self, frame, top_down_frame):
        frame = self.add_top_down(frame, top_down_frame)
        frame = self.add_stats_bar(frame)
        return frame

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
