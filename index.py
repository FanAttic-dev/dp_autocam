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
from video_recorder import VideoRecorder
import argparse

mousePos = {
    "x": 0,
    "y": 0
}


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        mousePos["x"] = x
        mousePos["y"] = y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--record", action='store_true')
    return parser.parse_args()


""" Init """
args = parse_args()

# video_path = get_random_file(videos_dir)
video_path = Path(
    "/home/atti/source/datasets/SoccerTrack/wide_view/videos/F_20200220_1_0120_0150.mp4")
player = VideoPlayer(video_path)
delay = player.get_delay(args.record)

pitch_coords = load_json(coords_path)
top_down = TopDown(pitch_coords)
detector = YoloPlayerDetector(pitch_coords)
ball_detector = YoloBallDetector(pitch_coords)

is_alive, frame_orig = player.get_next_frame()
camera = PerspectiveCamera(frame_orig)
frame_splitter = PerspectiveFrameSplitter(frame_orig)
# player.create_window("Original")
# cv2.setMouseCallback("Original", mouse_callback)

if args.record:
    recorder = VideoRecorder(player, camera.FRAME_W, camera.FRAME_H)

i = 0
while is_alive:
    is_alive, frame_orig = player.get_next_frame()
    if not is_alive:
        break

    h, w, _ = frame_orig.shape
    frame_orig = detector.preprocess(frame_orig)

    """ Detection """
    if not camera.pause_measurements:
        # Split frame, detect objects, merge & draw bounding boxes
        frames = frame_splitter.split(frame_orig)

        # bbs, _ = detector.detect(frames)
        bbs_ball, bbs_ball_frame = ball_detector.detect(frames)
        # for i, ball_frame in enumerate(bbs_ball_frame):
        #     player.show_frame(ball_frame, f"ball frame {i}")

        # bbs_joined = frame_splitter.join_bbs(bbs)
        bbs_joined = {
            "boxes": [],
            "cls": []
        }
        bbs_ball_joined = frame_splitter.join_bbs(bbs_ball)
        bb_ball = ball_detector.get_ball(bbs_ball_joined)
        add_bb_ball_(bbs_joined, bb_ball)
        detector.draw_bbs_(frame_orig, bbs_joined)
    else:
        bb_ball = []

    """ ROI """
    camera.update_by_bbs(bbs_joined, bb_ball, top_down)
    # camera.update_by_bbs([], bb_ball, top_down)
    frame = camera.get_frame(frame_orig)
    player.show_frame(frame, "ROI")
    # camera.print()
    # camera.draw_center_(frame_orig)
    # frame_splitter.draw_roi_(frame_orig)
    # camera.draw_roi_(frame_orig)
    # player.show_frame(frame_orig, "Original")

    """ Top-down """
    # top_down_frame = top_down.pitch_model.copy()
    # top_down.draw_roi_(top_down_frame, camera)

    # top_down_pts = top_down.bbs2points(bbs_joined)
    # top_down.draw_points_(top_down_frame, top_down_pts)

    # player.show_frame(top_down_frame, "top down")

    """ Warp frame """
    # frame_warped = top_down.warp_frame(frame_orig)
    # player.show_frame(frame_warped, "warped")

    """ Input """
    if args.record:
        recorder.write(frame)

    key = cv2.waitKey(delay)
    is_alive = camera.process_input(key, mousePos["x"], mousePos["y"])

    i += 1

print(f"Video: {video_path}")
if args.record:
    recorder.release()
player.release()
