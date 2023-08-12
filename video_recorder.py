from pathlib import Path
import cv2
from constants import colors


class VideoRecorder:
    RECORDINGS_FOLDER = Path("./recordings")
    FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
    SUFFIX = ".mp4"
    IS_COLOR = True
    STATS_WIDTH = 500
    TEXT_MARGIN = 20
    TEXT_FORMAT = {
        "fontFace": cv2.FONT_HERSHEY_DUPLEX,
        "fontScale": 1.0,
        "thickness": 1
    }

    def __init__(self, video_player, camera):
        self.camera = camera
        self.file_path = VideoRecorder.get_file_path(video_player)

        self.writer = cv2.VideoWriter(
            self.file_path_str,
            VideoRecorder.FOURCC,
            video_player.fps,
            self.frame_size,
            VideoRecorder.IS_COLOR
        )

        print(f"Video recorder initialized: {self.file_path_str}")

    @staticmethod
    def get_file_path(video_player):
        VideoRecorder.RECORDINGS_FOLDER.mkdir(exist_ok=True)

        video_path = video_player.video_path.stem
        file_path = VideoRecorder.RECORDINGS_FOLDER / video_path
        file_path = file_path.with_suffix(VideoRecorder.SUFFIX)

        idx = 1
        while file_path.exists():
            file_path = file_path.with_stem(f"{video_path}_{idx:02}")
            idx += 1

        return file_path

    @staticmethod
    def get_text_height():
        (_, h), _ = cv2.getTextSize(text="Lorem ipsum", **VideoRecorder.TEXT_FORMAT)
        return h

    @property
    def file_path_str(self):
        return str(self.file_path.absolute())

    @property
    def frame_size(self):
        return self.camera.FRAME_W + VideoRecorder.STATS_WIDTH, self.camera.FRAME_H

    @property
    def text_x(self):
        frame_w, _ = self.frame_size
        return frame_w - VideoRecorder.STATS_WIDTH + VideoRecorder.TEXT_MARGIN

    def add_stats_bar(self, frame):
        def add_border(frame):
            return cv2.copyMakeBorder(
                frame,
                0, 0, 0, VideoRecorder.STATS_WIDTH,
                borderType=cv2.BORDER_CONSTANT,
                value=0
            )

        def put_dict_items(frame, dict):
            spacing = VideoRecorder.get_text_height() + VideoRecorder.TEXT_MARGIN
            text_y = VideoRecorder.TEXT_MARGIN + spacing
            for key, value in dict.items():
                frame = cv2.putText(
                    img=frame,
                    text=f"{key}: {value}",
                    org=(self.text_x, text_y),
                    color=colors["white"],
                    **VideoRecorder.TEXT_FORMAT
                )
                text_y += spacing

        frame = add_border(frame)
        put_dict_items(frame, self.camera.model.get_stats())

        return frame

    def write(self, frame):
        frame = self.add_stats_bar(frame)
        self.writer.write(frame)
        return frame

    def release(self):
        self.writer.release()
