from pathlib import Path
import cv2
import os


class VideoRecorder:
    RECORDINGS_FOLDER = Path("./recordings")
    FOURCC = cv2.VideoWriter_fourcc(*'mp4v')
    SUFFIX = ".mp4"
    IS_COLOR = True

    def __init__(self, video_player, width, height):
        VideoRecorder.RECORDINGS_FOLDER.mkdir(exist_ok=True)
        self.file_path = VideoRecorder.get_file_path(video_player)
        self.frame_size = (width, height)

        self.writer = cv2.VideoWriter(
            self.file_path_str,
            VideoRecorder.FOURCC,
            video_player.fps,
            self.frame_size,
            VideoRecorder.IS_COLOR)
        print(f"Video recorder initialized: {self.file_path_str}")

    @staticmethod
    def get_file_path(video_player):
        video_path = video_player.video_path.stem
        file_path = VideoRecorder.RECORDINGS_FOLDER / video_path
        file_path = file_path.with_suffix(VideoRecorder.SUFFIX)

        idx = 1
        while file_path.exists():
            file_path = file_path.with_stem(f"{video_path}_{idx:02}")
            idx += 1

        return file_path

    @property
    def file_path_str(self):
        return str(self.file_path.absolute())

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()
