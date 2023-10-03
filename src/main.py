import cv2
import argparse
from camera.camera import PerspectiveCamera
from utils.config import Config
from detection.detector import YoloPlayerDetector
from detection.frame_splitter import FrameSplitter
from utils.profiler import Profiler
from camera.top_down import TopDown
from video_tools.video_player import VideoPlayer
from video_tools.video_recorder import VideoRecorder
from utils.constants import colors

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
config = Config(args)

player = VideoPlayer(config.video_path)
delay = player.get_delay(args.record)

is_alive, frame_orig = player.get_next_frame()
camera = PerspectiveCamera(frame_orig, config)
frame_splitter = FrameSplitter(frame_orig, config)
top_down = TopDown(config.pitch_coords, camera)
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
export_interval_sec = Config.autocam["eval"]["export_every_x_seconds"]
while is_alive:
    profiler = Profiler(frame_id)
    profiler.start("Total")

    profiler.start("Preprocess")
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
    if Config.autocam["drawing"]["enabled"]:
        detector.draw_bbs_(frame_orig, bbs_joined)

    if Config.autocam["drawing"]["show_split_frames"]:
        for i, bbs_frame in enumerate(bbs_frames):
            player.show_frame(bbs_frame, f"bbs_frame {i}")

    """ ROI """
    if args.mouse:
        camera.pid_x.update(mousePos["x"])
        camera.pid_y.update(mousePos["y"])
        pid_x = camera.pid_x.get()
        pid_y = camera.pid_y.get()
        camera.set_center(pid_x, pid_y)

        if Config.autocam["drawing"]["enabled"]:
            camera.draw_center_(frame_orig)
    else:
        camera.update_by_bbs(bbs_joined, top_down)

        if Config.autocam["drawing"]["enabled"]:
            camera.draw_ball_prediction_(frame_orig, colors["red"])
            camera.draw_ball_u_(frame_orig, colors["orange"])
            camera.ball_filter.draw_particles_(frame_orig)

    frame = camera.get_frame(frame_orig)
    if Config.autocam["drawing"]["enabled"] and Config.autocam["dead_zone"]["enabled"]:
        camera.draw_dead_zone_(frame)

    # if Config.params["debug"]:
    #     camera.print()

    """ Original frame """
    if Config.autocam["drawing"]["enabled"]:
        frame_splitter.draw_roi_(frame_orig)
        camera.draw_players_bb(frame_orig, bbs_joined)
        camera.draw_roi_(frame_orig)
    if not args.hide_windows and Config.autocam["drawing"]["show_original"]:
        player.show_frame(frame_orig, "Original")

    """ Top-down """
    top_down_frame = top_down.get_frame(bbs_joined)
    if not args.hide_windows and Config.autocam["drawing"]["show_top_down_window"]:
        player.show_frame(top_down_frame, "top down")

    """ Recorder """
    recorder_frame = recorder.get_frame(frame, top_down_frame)

    if not args.hide_windows:
        player.show_frame(recorder_frame, "ROI")

    if args.record:
        recorder.write(recorder_frame)

    """ Warp frame """
    frame_orig = camera.draw_frame_mask(frame_orig)
    frame_warped = top_down.warp_frame(
        frame_orig, overlay=Config.autocam["eval"]["pitch_overlay"])

    frame_sec = frame_id / int(player.fps)
    if args.record and Config.autocam["eval"]["export_enabled"] and \
            frame_sec % export_interval_sec == 0:
        frame_img_id = int(frame_sec // export_interval_sec)
        recorder.save_frame(frame, frame_img_id)
        recorder.save_frame(frame_warped, frame_img_id, "warped")

    if not args.hide_windows:
        player.show_frame(frame_warped, "warped")

    """ Profiler """
    profiler.stop("Other")
    profiler.stop("Total")
    if Config.autocam["verbose"]:
        profiler.print_summary()

    """ Next frame """
    is_alive, frame_orig = player.get_next_frame()
    frame_id += 1

    """ Input """
    key = cv2.waitKey(delay)
    is_alive = is_alive and camera.process_input(


print(f"Video: {config.video_path}")
recorder.release()
player.release()
