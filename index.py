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

frame_id = 0
while is_alive:
    t_frame_start = time.time()

    t_preprocess_start = time.time()
    is_alive, frame_orig = player.get_next_frame()
    if not is_alive:
        break

    h, w, _ = frame_orig.shape
    frame_orig_masked = detector.preprocess(frame_orig)
    # frame_orig = frame_orig_masked

    t_preprocess_elapsed = time.time() - t_preprocess_start

    """ Detection """
    bbs_joined = {
        "boxes": [],
        "cls": [],
        "ids": []
    }

    # Split
    t_split_start = time.time()
    frames = frame_splitter.split(frame_orig_masked)
    t_split_elapsed = time.time() - t_split_start
    # Detect
    t_detection_start = time.time()
    bbs, bbs_frames = detector.detect(frames)
    t_detection_elapsed = time.time() - t_detection_start
    # Join
    t_join_start = time.time()
    bbs_joined = frame_splitter.join_bbs(bbs)
    t_join_elapsed = time.time() - t_join_start

    t_other_start = time.time()

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
        camera.draw_roi_(frame_orig)
    if not args.hide_windows and params["drawing"]["show_original"]:
        player.show_frame(frame_orig, "Original")

    """ Top-down """
    top_down_frame = top_down.get_frame(bbs_joined)
    if not args.hide_windows and params["drawing"]["show_top_down_window"]:
        player.show_frame(top_down_frame, "top down")

    """ Warp frame """
    # frame_warped = top_down.warp_frame(frame_orig)
    # player.show_frame(frame_warped, "warped")

    """ Recorder """

    recorder_frame = recorder.get_frame(frame, top_down_frame)
    if not args.hide_windows:
        player.show_frame(recorder_frame, "ROI")

    if args.record:
        recorder.write(recorder_frame)

    t_other_elapsed = time.time() - t_other_start
    """ Timer """
    if params["verbose"]:
        frame_id += 1
        t_frame_elapsed = time.time() - t_frame_start
        print(f"""\
[Frame {frame_id:3}] \
Preprocess: {t_preprocess_elapsed*1000:3.0f}ms \
| Split: {t_split_elapsed*1000:3.0f}ms \
| Detect: {t_detection_elapsed*1000:3.0f}ms \
| Join: {t_join_elapsed*1000:3.0f}ms \
| Other: {t_other_elapsed*1000:3.0f}ms \
|| Total: {t_frame_elapsed:.2f}s ({1/t_frame_elapsed:.1f}fps)
""")

    """ Input """
    key = cv2.waitKey(delay)
    is_alive = camera.process_input(key, mousePos["x"], mousePos["y"])


print(f"Video: {video_path}")
recorder.release()
player.release()
