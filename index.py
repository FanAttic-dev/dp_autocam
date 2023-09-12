import json
import cv2
import argparse
from pathlib import Path
from camera import PerspectiveCamera
from constants import videos_dir, config_path, video_path, params
from detector import YoloBallDetector, YoloPlayerDetector
from frame_splitter import PerspectiveFrameSplitter
from utils import get_bbs_ball, get_random_file
from top_down import TopDown
from utils import load_json
from video_player import VideoPlayer
from video_recorder import VideoRecorder
from constants import colors

mousePos = {
    "x": 0,
    "y": 0
}


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        mousePos["x"] = x
        mousePos["y"] = y


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', "--record", action='store_true')
    parser.add_argument('-m', "--mouse", action='store_true')
    return parser.parse_args()


""" Init """
args = parse_args()

config = load_json(config_path)
pitch_coords = config["pitch_coords"]

player = VideoPlayer(video_path)
delay = player.get_delay(args.record)

is_alive, frame_orig = player.get_next_frame()
camera = PerspectiveCamera(frame_orig, config)
frame_splitter = PerspectiveFrameSplitter(frame_orig, config)
top_down = TopDown(pitch_coords, camera)
detector = YoloPlayerDetector(pitch_coords)
ball_detector = YoloBallDetector(pitch_coords, camera.ball_filter)

# args.record = True
# args.mouse = True

if args.mouse:
    player.create_window("Original")
    cv2.setMouseCallback("Original", mouse_callback)

recorder = VideoRecorder(player, camera, ball_detector)
if args.record:
    recorder.init_writer()

i = 0
while is_alive:
    is_alive, frame_orig = player.get_next_frame()
    if not is_alive:
        break

    h, w, _ = frame_orig.shape
    frame_orig_masked = detector.preprocess(frame_orig)
    # frame_orig = frame_orig_masked

    """ Detection """
    bbs_joined = {
        "boxes": [],
        "cls": [],
        "ids": []
    }
    bbs_ball_joined = {
        "boxes": [],
        "cls": [],
        "ids": []
    }
    if not camera.pause_measurements and not args.mouse:
        """ Split frame, detect objects, merge & draw bounding boxes """
        frames = frame_splitter.split(frame_orig_masked)

        # Players
        bbs, bbs_frames = detector.detect(frames)
        if params["show_split_frames"]["players"]:
            for i, bbs_frame in enumerate(bbs_frames):
                player.show_frame(bbs_frame, f"bbs_frame {i}")
        bbs_joined = frame_splitter.join_bbs(bbs)
        bbs_ball_joined, bbs_joined = get_bbs_ball(bbs_joined)
        detector.draw_bbs_(frame_orig, bbs_joined)

        # Balls
        # bbs_ball, bbs_ball_frames = ball_detector.detect(frames)
        # if params["show_split_frames"]["ball"]:
        #     for i, ball_frame in enumerate(bbs_ball_frames):
        #         player.show_frame(ball_frame, f"ball frame {i}")
        # bbs_ball_joined = frame_splitter.join_bbs(bbs_ball)

        detector.draw_bbs_(frame_orig, bbs_ball_joined, colors["white"])

    """ ROI """
    if args.mouse:
        camera.pid_x.update(mousePos["x"])
        camera.pid_y.update(mousePos["y"])
        pid_x = camera.pid_x.get()
        pid_y = camera.pid_y.get()
        camera.set_center(pid_x, pid_y)

        camera.draw_center_(frame_orig)
    else:
        camera.update_by_bbs(bbs_joined, bbs_ball_joined, top_down)
        camera.draw_ball_prediction_(frame_orig, colors["red"])
        camera.draw_ball_u_(frame_orig, colors["orange"])
        camera.ball_filter.draw_particles_(frame_orig)
        ...

    frame = camera.get_frame(frame_orig)

    # camera.draw_dead_zone_(frame)
    # player.show_frame(frame, "ROI")
    # camera.print()

    # frame_splitter.draw_roi_(frame_orig)
    # camera.draw_roi_(frame_orig)
    # player.show_frame(frame_orig, "Original")

    """ Top-down """
    top_down_frame = top_down.get_frame(bbs_joined)
    # player.show_frame(top_down_frame, "top down")

    """ Warp frame """
    # frame_warped = top_down.warp_frame(frame_orig)
    # player.show_frame(frame_warped, "warped")

    """ Recorder """
    recorder_frame = recorder.get_frame(frame, top_down_frame)
    player.show_frame(recorder_frame, "ROI")
    if args.record:
        recorder.write(recorder_frame)

    """ Input """
    key = cv2.waitKey(delay)
    is_alive = camera.process_input(key, mousePos["x"], mousePos["y"])

    i += 1

print(f"Video: {video_path}")
recorder.release()
player.release()
