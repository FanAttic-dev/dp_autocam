from pathlib import Path
import cv2
from utils.config import Config

from video_tools.video_player import VideoPlayer

videos_dir = Path("../../datasets/TrnavaZilina/VAR")
im_size = (1280, 720)
export_interval_sec = Config.autocam["eval"]["export_every_x_seconds"]


def get_frames_dir(videos_dir, video_path):
    return videos_dir / video_path.with_stem(f"{video_path.stem}_frames").stem


def get_frame_path(frame_id):
    return frames_dir / f"{video_path.stem}_frame_{frame_id:04d}.jpg"


for period in ["var_p0.mp4", "var_p1.mp4"]:
    video_path = videos_dir / period
    frames_dir = get_frames_dir(videos_dir, video_path)
    Path.mkdir(frames_dir, exist_ok=False)

    player = VideoPlayer(video_path)

    is_alive = True
    frame_id = 0
    while is_alive:
        # is_alive, frame = player.get_next_frame()
        is_alive = player.cap.grab()

        frame_sec = frame_id / int(player.fps)
        if frame_sec % export_interval_sec == 0:
            frame_img_id = int(frame_sec // export_interval_sec)

            frame_path = get_frame_path(frame_img_id)
            print(str(frame_path))

            is_alive, frame = player.cap.retrieve()
            frame = cv2.resize(frame, im_size)
            cv2.imwrite(str(frame_path), frame)

        frame_id += 1

    player.release()
