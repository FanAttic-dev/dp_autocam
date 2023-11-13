import cv2

from camera.rectilinear_camera import RectilinearCamera
from cameraman.autocam_cameraman import AutocamCameraman
from utils.argparse import AutocamArgsNamespace, parse_args
from utils.config import Config
from detection.yolo_detector import YoloBallDetector, YoloDetector
from camera.frame_splitter import FrameSplitter
from utils.profiler import Profiler
from camera.top_down import TopDown
from video_tools.video_player import VideoPlayer
from video_tools.video_recorder import VideoRecorder
from utils.constants import Color
import utils.utils as utils


class Autocam:
    def __init__(self, args: AutocamArgsNamespace, config: Config):
        self.args = args
        self.config = config
        self.is_debug = not self.args.no_debug or self.args.mouse

        self.player = VideoPlayer(self.config.video_path)
        self.delay = self.player.get_delay(self.args.record)

        _, frame_orig = self.player.get_next_frame()
        self.player.restart()
        self.camera = RectilinearCamera(frame_orig, self.config)
        self.frame_splitter = FrameSplitter(frame_orig, self.config)
        self.top_down = TopDown(self.config.pitch_corners, self.camera)
        self.detector = YoloDetector(
            frame_orig, self.top_down, self.config)
        self.cameraman = AutocamCameraman(
            self.camera, self.top_down, self.config)

        if Config.autocam["detector"]["ball"]["enabled"]:
            self.ball_detector = YoloBallDetector(
                frame_orig, self.top_down, self.config)

        if self.args.mouse:
            self.player.init_mouse("Original")

        self.recorder = VideoRecorder(
            self.config, self.player, self.camera, self.detector, self.cameraman)
        if self.args.record:
            self.recorder.init_writer()
        if self.args.record and self.is_debug:
            self.recorder.init_debug_writer()

    def run(self):
        is_alive, frame_orig = self.player.get_next_frame()
        frame_id = 0
        while is_alive:
            self.process_frame(frame_orig, frame_id)

            """ Next frame """
            is_alive, frame_orig = self.player.get_next_frame()
            frame_id += 1

            """ Input """
            key = cv2.waitKey(self.delay)
            is_alive = is_alive and self.process_input(key)

        self.finish()

    def process_input(self, key) -> bool:
        if key == ord('m'):
            self.args.mouse = not self.args.mouse

        return self.camera.process_input(
            key, self.player.mouse_pos
        )

    def process_frame(self, frame_orig, frame_id: int):
        profiler = Profiler(frame_id)
        profiler.start("Total")

        profiler.start("Preprocess")
        frame_orig_masked = self.detector.preprocess(frame_orig)
        if self.is_debug and Config.autocam["debug"]["show_frame_mask"]:
            frame_orig = frame_orig_masked
        if self.is_debug:
            frame_orig_debug = frame_orig.copy()
        profiler.stop("Preprocess")

        """ Detection """
        bbs_joined = {
            "boxes": [],
            "cls": [],
            "ids": []
        }

        if not self.args.mouse and Config.autocam["detector"]["enabled"]:
            # Split
            profiler.start("Split")
            frames = self.frame_splitter.split(frame_orig_masked)
            profiler.stop("Split")
            # Detect
            profiler.start("Detect")
            bbs, bbs_frames = self.detector.detect(frames)
            if Config.autocam["detector"]["ball"]["enabled"]:
                bbs_ball, _ = self.ball_detector.detect(frames)
            profiler.stop("Detect")
            # Join
            profiler.start("Join")
            bbs_joined = self.frame_splitter.flatten_bbs(bbs)
            if Config.autocam["detector"]["ball"]["enabled"]:
                bbs_ball = self.frame_splitter.flatten_bbs(bbs_ball)
                bbs_joined = utils.join_bbs(bbs_joined, bbs_ball)

            if Config.autocam["detector"]["filter_detections"]:
                self.detector.filter_detections_(bbs_joined)
            profiler.stop("Join")

            if self.is_debug and not self.args.hide_windows and Config.autocam["debug"]["show_split_frames"]:
                for i, bbs_frame in enumerate(bbs_frames):
                    self.player.show_frame(bbs_frame, f"bbs_frame {i}")

        """ ROI """
        if self.is_debug and self.args.mouse:
            if Config.autocam["debug"]["mouse_use_pid"]:
                self.cameraman._try_update_camera(self.player.mouse_pos)
            self.camera.draw_center_(frame_orig_debug)
        else:
            profiler.start("Update by BBS")
            self.cameraman.update_camera(bbs_joined)
            profiler.stop("Update by BBS")
            self.camera.draw_center_(frame_orig_debug)

        profiler.start("Get ROI")
        frame_roi = self.camera.get_frame_roi(frame_orig)
        profiler.stop("Get ROI")

        profiler.start("Other")
        if not self.args.mouse and self.is_debug and Config.autocam["detector"]["enabled"]:
            if Config.autocam["debug"]["draw_detections"]:
                self.detector.draw_bbs_(frame_orig_debug, bbs_joined)
                self.cameraman.draw_ball_prediction_(
                    frame_orig_debug, Color.RED)
                self.cameraman.draw_ball_u_(frame_orig_debug, Color.ORANGE)
                self.cameraman.ball_filter.draw_particles_(frame_orig_debug)
            if Config.autocam["debug"]["draw_players_bb"]:
                self.cameraman.draw_players_bb_(frame_orig_debug, bbs_joined)
        if self.is_debug:
            frame_roi_debug = self.camera.get_frame_roi(frame_orig_debug)

        if self.is_debug and Config.autocam["debug"]["print_camera_stats"]:
            self.camera.print()

        """ Original frame """
        if self.is_debug and Config.autocam["debug"]["draw_roi"]:
            self.frame_splitter.draw_roi_(frame_orig_debug)
            self.camera.draw_roi_(frame_orig_debug)
        if self.is_debug and Config.autocam["debug"]["draw_grid"]:
            self.camera.draw_grid_(frame_orig_debug)
        if not self.args.hide_windows and Config.autocam["show_original"]:
            self.player.show_frame(
                frame_orig_debug if self.is_debug else frame_orig, "Original")

        """ Top-down """
        draw_players_center = self.is_debug and \
            Config.autocam["debug"]["draw_top_down_players_center"]
        players_center = self.cameraman.players_filter.pos if draw_players_center else None

        frame_top_down = self.top_down.get_frame(
            bbs_joined, players_center
        )

        if not self.args.hide_windows and Config.autocam["show_top_down_window"]:
            self.player.show_frame(frame_top_down, "top down")

        """ Recorder """
        if self.is_debug:
            frame_roi_debug = self.recorder.decorate_frame(
                frame_roi_debug, frame_top_down
            )

        if not self.args.hide_windows:
            self.player.show_frame(
                frame_roi_debug if self.is_debug else frame_roi, "ROI"
            )

        if self.args.record:
            self.recorder.write(frame_roi)
            if self.is_debug:
                self.recorder.write_debug(frame_roi_debug)

        """ Warp frame """
        frame_orig_masked = self.camera.draw_frame_mask(frame_orig)
        frame_orig_warped = self.top_down.warp_frame(
            frame_orig_masked,
            overlay=Config.autocam["eval"]["pitch_overlay"]
        )

        frame_sec = frame_id / int(self.player.fps)
        if self.args.record and self.args.export_interval_sec > -1 and \
                frame_sec % self.args.export_interval_sec == 0:
            frame_img_id = int(frame_sec // self.args.export_interval_sec)
            self.recorder.save_frame(frame_roi, frame_img_id)
            self.recorder.save_frame(frame_orig_warped, frame_img_id, "warped")

        if not self.args.hide_windows and self.is_debug and Config.autocam["debug"]["show_warped_frame"]:
            self.player.show_frame(frame_orig_warped, "warped")

        """ Profiler """
        profiler.stop("Other")
        profiler.stop("Total")
        if Config.autocam["print_profiler_stats"]:
            profiler.print_summary()

    def finish(self):
        print(f"Video: {self.config.video_path}")
        self.recorder.release()
        self.player.release()
