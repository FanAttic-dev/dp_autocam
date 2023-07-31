from pathlib import Path
import cv2
from camera import PerspectiveCamera
from constants import videos_dir, coords_path
from detector import YoloBallDetector, YoloPlayerDetector
from frame_splitter import PerspectiveFrameSplitter
from utils import add_bb_, add_bb_ball_, get_random_file
from top_down import TopDown
from utils import load_json
from video_player import VideoPlayer

""" Init """
video_name = get_random_file(videos_dir)
video_name = Path(
    "/home/atti/source/datasets/SoccerTrack/wide_view/videos/F_20220220_1_1530_1560.mp4")
print(f"Video: {video_name}")
player = VideoPlayer(video_name)

pitch_coords = load_json(coords_path)
top_down = TopDown(pitch_coords)
detector = YoloPlayerDetector(pitch_coords)
ball_detector = YoloBallDetector(pitch_coords)

is_alive, frame_orig = player.get_next_frame()
camera = PerspectiveCamera(frame_orig)
frame_splitter = PerspectiveFrameSplitter(frame_orig)

i = 0
while is_alive:
    is_alive, frame_orig = player.get_next_frame()
    if not is_alive:
        break

    h, w, _ = frame_orig.shape
    frame_orig = detector.preprocess(frame_orig)

    """ Detection """
    # Split frame, detect objects, merge & draw bounding boxes
    frames = frame_splitter.split(frame_orig)
    frame_splitter.draw_roi_(frame_orig)

    bbs, _ = detector.detect(frames)
    bbs_ball, bbs_ball_frame = ball_detector.detect(frames)
    player.show_frame(bbs_ball_frame[0])

    bbs_joined = frame_splitter.join_bbs(bbs)
    bbs_ball_joined = frame_splitter.join_bbs(bbs_ball)
    bb_ball = ball_detector.get_ball(bbs_ball_joined)
    add_bb_ball_(bbs_joined, bb_ball)
    # bbs_joined = merge_bbs(bbs_joined, bbs_ball_joined)
    detector.draw_bbs_(frame_orig, bbs_joined)

    """ ROI """
    camera.update_by_bbs(bbs_joined, bb_ball)
    # frame = camera.get_frame(frame_orig)
    # player.show_frame(frame, "ROI")
    camera.print()
    camera.draw_roi_(frame_orig)
    player.show_frame(frame_orig, "Original")

    """ Top-down """
    top_down_frame = top_down.pitch_model.copy()
    top_down.draw_roi_(top_down_frame, camera)

    top_down_pts = top_down.bbs2points(bbs_joined)
    top_down.draw_points_(top_down_frame, top_down_pts)

    player.show_frame(top_down_frame, "top down")

    """ Warp frame """
    # frame_warped = top_down.warp_frame(frame_orig)
    # player.show_frame(frame_warped, "warped")

    """ Input """
    key = cv2.waitKey(0)
    is_alive = camera.process_input(key)

    i += 1

print(f"Video: {video_name}")
player.release()
