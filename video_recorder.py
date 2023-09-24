from functools import cached_property
from pathlib import Path
import cv2
from constants import colors
from utils import path2str


class VideoRecorder:
    RECORDINGS_FOLDER = Path("./recordings")
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

    def __init__(self, video_player, camera, detector):
        self.camera = camera
        self.video_player = video_player
        self.detector = detector
        self.file_path = VideoRecorder.get_file_path(video_player)
        self.writer = None

    def init_writer(self):
        self.writer = cv2.VideoWriter(
            path2str(self.file_path),
            VideoRecorder.FOURCC,
            self.video_player.fps,
            self.frame_size,
            VideoRecorder.IS_COLOR
        )

        print(f"Video recorder initialized: {path2str(self.file_path)}")

    @staticmethod
    def get_file_path(video_player):
        VideoRecorder.RECORDINGS_FOLDER.mkdir(exist_ok=True)

        video_path = video_player.video_path.stem
        file_path = VideoRecorder.RECORDINGS_FOLDER / video_path
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
        return self.camera.FRAME_W + VideoRecorder.STATS_WIDTH, self.camera.FRAME_H

    @cached_property
    def text_x(self):
        frame_w, _ = self.frame_size
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
                    color=colors["white"],
                    **VideoRecorder.TEXT_FORMAT
                )
                text_y += self.spacing
            text_y += self.spacing

        def get_stats(model, name):
            stats = model.get_stats()
            stats["Name"] = f"{name}: {stats['Name']}"
            return stats

        detector_stats = get_stats(self.detector, "Detector")
        camera_stats = get_stats(self.camera, "Camera")
        pid_x_stats = get_stats(self.camera.pid_x, "PID_X")
        pid_y_stats = get_stats(self.camera.pid_y, "PID_Y")
        pid_f_stats = get_stats(self.camera.pid_f, "PID_F")
        ball_stats = get_stats(self.camera.ball_filter, "Ball")

        text_y = self.spacing

        frame = add_border(frame)

        # put_dict_items_(frame, detector_stats)
        put_dict_items_(frame, camera_stats)
        # put_dict_items_(frame, pid_x_stats)
        # put_dict_items_(frame, pid_y_stats)
        put_dict_items_(frame, pid_f_stats)
        put_dict_items_(frame, ball_stats)

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

    def save_frame(self, frame, frame_id, suffix=""):
        dir_path = self.file_path.parent / f"{self.file_path.stem}_frames"
        Path.mkdir(dir_path, exist_ok=True)

        filename = f"{self.file_path.stem}_{frame_id:04d}"
        if suffix:
            filename += f"_{suffix}"
        filename += VideoRecorder.IMG_SUFFIX

        filepath = dir_path / filename
        cv2.imwrite(path2str(filepath), frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
