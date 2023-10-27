import cv2
from algorithm.autocam_algo import AutocamAlgo
from camera.rectilinear_camera import RectilinearCamera
from utils.argparse import parse_args
from utils.config import Config
from detection.yolo_detector import YoloBallDetector, YoloDetector, YoloPlayerDetector
from detection.frame_splitter import FrameSplitter
from utils.profiler import Profiler
from camera.top_down import TopDown
from video_tools.video_player import VideoPlayer
from video_tools.video_recorder import VideoRecorder
from utils.constants import Color
import utils.utils as utils


""" Init """
args = parse_args()
# args.record = True
# args.mouse = True
config = Config(args)
is_debug = not args.no_debug or args.mouse

player = VideoPlayer(config.video_path)
delay = player.get_delay(args.record)

is_alive, frame_orig = player.get_next_frame()
camera = RectilinearCamera(frame_orig, config)
frame_splitter = FrameSplitter(frame_orig, config)
top_down = TopDown(config.pitch_corners, camera)
ball_detector = YoloBallDetector(frame_orig, top_down, config)
detector = YoloPlayerDetector(frame_orig, top_down, config)
algo = AutocamAlgo(camera, top_down, config)

if args.mouse:
    player.init_mouse("Original")

recorder = VideoRecorder(player, camera, detector, algo)
if args.record:
    recorder.init_writer()
    if is_debug:
        recorder.init_debug_writer()

frame_id = 0
export_interval_sec = Config.autocam["eval"]["export_every_x_seconds"]
while is_alive:
    profiler = Profiler(frame_id)
    profiler.start("Total")

    profiler.start("Preprocess")
    frame_orig_masked = detector.preprocess(frame_orig)
    if is_debug and Config.autocam["debug"]["show_frame_mask"]:
        frame_orig = frame_orig_masked
    if is_debug:
        frame_orig_debug = frame_orig.copy()
    profiler.stop("Preprocess")

    """ Detection """
    bbs_joined = {
        "boxes": [],
        "cls": [],
        "ids": []
    }

    if not args.mouse and Config.autocam["detector"]["enabled"]:
        # Split
        profiler.start("Split")
        frames = frame_splitter.split(frame_orig_masked)
        profiler.stop("Split")
        # Detect
        profiler.start("Detect")
        bbs_player, bbs_frames = detector.detect(frames)
        bbs_ball, _ = ball_detector.detect(frames)
        profiler.stop("Detect")
        # Join
        profiler.start("Join")
        bbs_player = frame_splitter.flatten_bbs(bbs_player)
        bbs_ball = frame_splitter.flatten_bbs(bbs_ball)
        bbs_joined = utils.join_bbs(bbs_player, bbs_ball)

        if Config.autocam["detector"]["filter_detections"]:
            detector.filter_detections_(bbs_joined)
        profiler.stop("Join")

        if is_debug and not args.hide_windows and Config.autocam["debug"]["show_split_frames"]:
            for i, bbs_frame in enumerate(bbs_frames):
                player.show_frame(bbs_frame, f"bbs_frame {i}")

    """ ROI """
    if is_debug and args.mouse:
        if config.autocam["debug"]["mouse_use_pid"]:
            algo.try_update_camera(player.mouse_pos)
        camera.draw_center_(frame_orig_debug)
    else:
        profiler.start("Update by BBS")
        algo.update_by_bbs(bbs_joined)
        profiler.stop("Update by BBS")

    profiler.start("Get frame")
    frame = camera.get_frame(frame_orig)
    profiler.stop("Get frame")

    profiler.start("Other")
    if not args.mouse and is_debug and Config.autocam["detector"]["enabled"]:
        if Config.autocam["debug"]["draw_detections"]:
            detector.draw_bbs_(frame_orig_debug, bbs_joined)
            algo.draw_ball_prediction_(frame_orig_debug, Color.RED)
            algo.draw_ball_u_(frame_orig_debug, Color.ORANGE)
            algo.ball_filter.draw_particles_(frame_orig_debug)
        if Config.autocam["debug"]["draw_players_bb"]:
            algo.draw_players_bb_(frame_orig_debug, bbs_joined)
    if is_debug:
        frame_debug = camera.get_frame(frame_orig_debug)

    if is_debug and Config.autocam["debug"]["print_camera_stats"]:
        camera.print()

    """ Original frame """
    if is_debug and Config.autocam["debug"]["draw_roi"]:
        frame_splitter.draw_roi_(frame_orig_debug)
        camera.draw_roi_(frame_orig_debug)
    if is_debug and Config.autocam["debug"]["draw_grid"]:
        camera.draw_grid_(frame_orig_debug)
    if not args.hide_windows and Config.autocam["show_original"]:
        player.show_frame(
            frame_orig_debug if is_debug else frame_orig, "Original")

    """ Top-down """
    draw_players_center = is_debug and Config.autocam["debug"]["draw_top_down_players_center"]
    players_center = algo.players_filter.pos if draw_players_center else None

    top_down_frame = top_down.get_frame(bbs_joined, players_center)

    if not args.hide_windows and Config.autocam["show_top_down_window"]:
        player.show_frame(top_down_frame, "top down")

    """ Recorder """
    if is_debug:
        frame_debug = recorder.decorate_frame(frame_debug, top_down_frame)

    if not args.hide_windows:
        player.show_frame(frame_debug if is_debug else frame, "ROI")

    if args.record:
        recorder.write(frame)
        if is_debug:
            recorder.write_debug(frame_debug)

    """ Warp frame """
    frame_orig_masked = camera.draw_frame_mask(frame_orig)
    frame_warped = top_down.warp_frame(
        frame_orig_masked,
        overlay=Config.autocam["eval"]["pitch_overlay"]
    )

    frame_sec = frame_id / int(player.fps)
    if args.record and args.export_frames and \
            frame_sec % export_interval_sec == 0:
        frame_img_id = int(frame_sec // export_interval_sec)
        recorder.save_frame(frame, frame_img_id)
        recorder.save_frame(frame_warped, frame_img_id, "warped")

    if not args.hide_windows and is_debug and Config.autocam["debug"]["show_warped_frame"]:
        player.show_frame(frame_warped, "warped")

    """ Profiler """
    profiler.stop("Other")
    profiler.stop("Total")
    if Config.autocam["print_profiler_stats"]:
        profiler.print_summary()

    """ Next frame """
    is_alive, frame_orig = player.get_next_frame()
    frame_id += 1

    """ Input """
    key = cv2.waitKey(delay)
    is_alive = is_alive and camera.process_input(
        key, player.mouse_pos
    )


print(f"Video: {config.video_path}")
recorder.release()
player.release()
