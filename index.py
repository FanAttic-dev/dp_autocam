from pathlib import Path
import cv2
from camera import FixedHeightCamera, PerspectiveCamera
from constants import PAN_DX, TILT_DY, ZOOM_DZ, videos_dir
import random
from detector import BgDetector, YoloPlayerDetector
from frame_splitter import FrameSplitter, LinearFrameSplitter, PerspectiveFrameSplitter
from utils import colors, get_random_file

from top_down import TopDown
from utils import load_json
from video_player import VideoPlayer

coords_path = videos_dir / "../../coords.json"
video_name = get_random_file(videos_dir)
# video_name = videos_dir / "F_20200220_1_0180_0210.mp4"
# video_name = videos_dir / "F_20200220_1_0480_0510.mp4"
# video_name = videos_dir / "F_20220220_1_1920_1950.mp4"
print(f"Video: {video_name}")
player = VideoPlayer(video_name)

pitch_coords = load_json(coords_path)
top_down = TopDown(pitch_coords)
detector = YoloPlayerDetector(pitch_coords)

ret, frame_orig = player.get_next_frame()
camera = PerspectiveCamera(frame_orig)
frame_splitter = PerspectiveFrameSplitter(frame_orig)

i = 0
while True:
    ret, frame_orig = player.get_next_frame()
    if not ret:
        break

    frame_orig = detector.preprocess(frame_orig)
    h, w, _ = frame_orig.shape

    # Split, detect & merge
    frames = frame_splitter.split(frame_orig)
    # frame_joined = frame_splitter.join(frames)

    bbs, frames_detected = detector.detect(frames)
    # for i, frame in enumerate(frames_detected):
    # frame_splitter.cameras[i].draw_roi_(frame_orig, color=colors[i])
    # show_frame(frame, f"Frame {i}")

    bbs_joined = frame_splitter.join_bbs(bbs)
    detector.draw_bounding_boxes_(frame_orig, bbs_joined)

    # frame_warped = top_down.warp_frame(frame_orig)
    # show_frame(frame_warped, "warped")

    # camera.update_by_bbs(bbs)
    frame = camera.get_frame(frame_orig)
    player.show_frame(frame, "ROI")
    camera.print()

    bb_pts = top_down.warp_bbs(bbs_joined)
    top_down_frame = top_down.pitch_model.copy()
    top_down.draw_roi_(top_down_frame, camera.get_corner_pts())
    top_down.draw_points_(top_down_frame, bb_pts)
    player.show_frame(top_down_frame, "top down")

    camera.draw_roi_(frame_orig)
    player.show_frame(frame_orig, "Original")

    key = cv2.waitKey(0)
    if key == ord('d'):
        camera.pan(PAN_DX)
    elif key == ord('a'):
        camera.pan(-PAN_DX)
    elif key == ord('w'):
        camera.tilt(-TILT_DY)
    elif key == ord('s'):
        camera.tilt(TILT_DY)
    elif key == ord('p'):
        camera.zoom(ZOOM_DZ)
    elif key == ord('m'):
        camera.zoom(-ZOOM_DZ)
    elif key == ord('r'):
        camera.reset()
    elif key == ord('q'):
        break

    i += 1

print(f"Video: {video_name}")
player.release()
