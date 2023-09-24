import json
import cv2
import argparse
from pathlib import Path
from camera import PerspectiveCamera
from constants import videos_dir, config_path, video_path, params
from detector import YoloBallDetector, YoloPlayerDetector
from frame_splitter import PerspectiveFrameSplitter
from profiler import Profiler
from utils import get_bbs_ball, get_bounding_box, get_random_file
from top_down import TopDown
from utils import load_json
from video_player import VideoPlayer
from video_recorder import VideoRecorder
from constants import colors
import time

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
    parser.add_argument('-v', "--video-path", action='store', required=False)
    parser.add_argument("--config-path", action='store', required=False)
    parser.add_argument("--hide-windows", action='store_true', default=False)
    return parser.parse_args()


""" Init """
args = parse_args()
if args.video_path:
    video_path = Path(args.video_path)
if args.config_path:
    config_path = Path(args.config_path)

config = load_json(config_path)
pitch_coords = config["pitch_coords"]

player = VideoPlayer(video_path)
delay = player.get_delay(args.record)

is_alive, frame_orig = player.get_next_frame()
camera = PerspectiveCamera(frame_orig, config)
frame_splitter = PerspectiveFrameSplitter(frame_orig, config)
top_down = TopDown(pitch_coords, camera)
detector = YoloPlayerDetector(frame_orig, top_down, config)

# args.record = True
# args.mouse = True

if args.mouse:
    player.create_window("Original")
    cv2.setMouseCallback("Original", mouse_callback)

recorder = VideoRecorder(player, camera, detector)
if args.record:
    recorder.init_writer()

frame_id = 0
while is_alive:
    profiler = Profiler(frame_id)
    profiler.start("Frame")

    profiler.start("Preprocess")
    is_alive, frame_orig = player.get_next_frame()
    if not is_alive:
        break

    frame_orig_masked = detector.preprocess(frame_orig)
    # frame_orig = frame_orig_masked

    profiler.stop("Preprocess")

    """ Detection """
    bbs_joined = {
        "boxes": [],
        "cls": [],
        "ids": []
    }

    # Split
    profiler.start("Split")
    frames = frame_splitter.split(frame_orig_masked)
    profiler.stop("Split")
    # Detect
    profiler.start("Detect")
    bbs, bbs_frames = detector.detect(frames)
    profiler.stop("Detect")
    # Join
    profiler.start("Join")
    bbs_joined = frame_splitter.join_bbs(bbs)
    profiler.stop("Detect")

    profiler.start("Other")

    # Render
    if params["drawing"]["enabled"]:
        detector.draw_bbs_(frame_orig, bbs_joined)

    if params["drawing"]["show_split_frames"]:
        for i, bbs_frame in enumerate(bbs_frames):
            player.show_frame(bbs_frame, f"bbs_frame {i}")

    """ ROI """
    if args.mouse:
        camera.pid_x.update(mousePos["x"])
        camera.pid_y.update(mousePos["y"])
        pid_x = camera.pid_x.get()
        pid_y = camera.pid_y.get()
        camera.set_center(pid_x, pid_y)

        if params["drawing"]["enabled"]:
            camera.draw_center_(frame_orig)
    else:
        camera.update_by_bbs(bbs_joined, top_down)

        if params["drawing"]["enabled"]:
            camera.draw_ball_prediction_(frame_orig, colors["red"])
            camera.draw_ball_u_(frame_orig, colors["orange"])
            camera.ball_filter.draw_particles_(frame_orig)

    frame = camera.get_frame(frame_orig)
    if params["drawing"]["enabled"] and params["dead_zone"]["enabled"]:
        camera.draw_dead_zone_(frame)

    if params["verbose"]:
        camera.print()

    """ Original frame """
    if params["drawing"]["enabled"]:
        frame_splitter.draw_roi_(frame_orig)
        camera.draw_players_bb(frame_orig, bbs_joined)
        camera.draw_roi_(frame_orig)
    if not args.hide_windows and params["drawing"]["show_original"]:
        player.show_frame(frame_orig, "Original")

    """ Top-down """
    top_down_frame = top_down.get_frame(bbs_joined)
    if not args.hide_windows and params["drawing"]["show_top_down_window"]:
        player.show_frame(top_down_frame, "top down")

    """ Recorder """
    recorder_frame = recorder.get_frame(frame, top_down_frame)
    if not args.hide_windows:
        player.show_frame(recorder_frame, "ROI")

    if args.record:
        recorder.write(recorder_frame)

    """ Warp frame """
    if not args.hide_windows:
        frame_orig = camera.draw_frame_mask(frame_orig)
        frame_warped = top_down.warp_frame(frame_orig)
        player.show_frame(frame_warped, "warped")

    """ Profiler """
    profiler.stop("Other")
    profiler.stop("Frame")
    if params["verbose"]:
        profiler.print_summary()

    """ Input """
    key = cv2.waitKey(delay)
    is_alive = camera.process_input(key, mousePos["x"], mousePos["y"])
    if key == ord('t'):
        recorder.save_img(frame_warped, frame_id)

    frame_id += 1


print(f"Video: {video_path}")
recorder.release()
player.release()
