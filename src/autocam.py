import cv2

from camera.rectilinear_camera import RectilinearCamera
from cameraman.autocam_cameraman import AutocamCameraman
from utils.argparse import AutocamArgsNamespace
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
        self.detector_enabled = not self.args.mouse and Config.autocam["detector"]["enabled"]

        self.profiler = Profiler().init(0)
        self.profiler.start("Init")

        self.player = VideoPlayer(self.config.video_path)
        self.delay = self.player.get_delay(self.args.record)

        _, frame_orig = self.player.get_next_frame()
        self.camera = RectilinearCamera(frame_orig, self.config)
        self.frame_splitter = FrameSplitter(frame_orig, self.config)
        self.top_down = TopDown(self.config.pitch_corners, self.camera)
        self.detector = YoloDetector(
            frame_orig, self.top_down, self.config
        )
        if Config.autocam["detector"]["ball"]["enabled"]:
            self.ball_detector = YoloBallDetector(
                frame_orig, self.top_down, self.config)

        self.cameraman = AutocamCameraman(
            self.camera, self.top_down, self.config
        )
        self.init_cameraman_filters(frame_orig)

        if self.args.mouse:
            self.player.init_mouse("Original")

        self.recorder = VideoRecorder(
            self.config, self.player, self.camera, self.detector, self.cameraman)
        if self.args.record:
            self.recorder.init_writer()
        if self.args.record and self.is_debug:
            self.recorder.init_debug_writer()

        self.profiler.stop("Init")
        self.profiler.print_summary()

    def init_cameraman_filters(self, frame_orig):
        """Set the cameraman filters to players center for initialization."""
        if not self.detector_enabled:
            return

        frame_orig, frame_orig_masked, _ = self.preprocess_frame(frame_orig)
        bbs = self.detect(frame_orig_masked)

        self.cameraman.init_filters_pos(bbs)

    def run(self):
        self.player.restart()  # go to the beginning of the video
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

    def process_frame(self, frame_orig, frame_id: int):
        self.profiler.init(frame_id)
        self.profiler.start("Total")

        # Preprocessing
        frame_orig, frame_orig_masked, frame_orig_debug = self.preprocess_frame(
            frame_orig
        )

        # Detection
        bbs_joined = self.detect(frame_orig_masked)

        # Update Camera
        frame_roi = self.update_camera(
            bbs_joined, frame_orig, frame_orig_debug)

        self.profiler.start("Other")

        # Visualize
        frame_roi_debug = self.draw_debug_info(frame_orig_debug, bbs_joined)

        # Original Frame
        self.show_original(frame_orig, frame_orig_debug)

        # TopDown Frame
        frame_top_down = self.show_top_down(bbs_joined)

        # ROI
        frame_roi_debug = self.decorate_roi(frame_roi_debug, frame_top_down)
        self.show_roi(frame_roi, frame_roi_debug)

        # Record
        self.write_roi(frame_roi, frame_roi_debug)

        # Warp
        frame_orig_warped = self.warp_frame(frame_orig)
        self.show_warped(frame_orig_warped)

        # Eval
        self.export_for_eval(frame_orig_warped, frame_roi, frame_id)

        # Stats
        self.profiler.stop("Other")
        self.profiler.stop("Total")
        self.print_stats()

    def preprocess_frame(self, frame_orig):
        self.profiler.start("Preprocess")
        frame_orig_masked = self.detector.preprocess(frame_orig)
        if self.is_debug and Config.autocam["debug"]["show_frame_mask"]:
            frame_orig = frame_orig_masked
        self.profiler.stop("Preprocess")
        return frame_orig, frame_orig_masked, (frame_orig.copy() if self.is_debug else None)

    def detect(self, frame_orig_masked):
        if not self.detector_enabled:
            return None

        # Split
        self.profiler.start("Split")
        frames = self.frame_splitter.split(frame_orig_masked)
        self.profiler.stop("Split")

        # Detect
        self.profiler.start("Detect")
        bbs, bbs_frames = self.detector.detect(frames)
        if Config.autocam["detector"]["ball"]["enabled"]:
            bbs_ball, _ = self.ball_detector.detect(frames)
        self.profiler.stop("Detect")

        # Join
        self.profiler.start("Join")
        bbs_joined = self.frame_splitter.flatten_bbs(bbs)
        if Config.autocam["detector"]["ball"]["enabled"]:
            bbs_ball = self.frame_splitter.flatten_bbs(bbs_ball)
            bbs_joined = utils.join_bbs(bbs_joined, bbs_ball)

        if Config.autocam["detector"]["filter_detections"]:
            self.detector.filter_detections_(bbs_joined)
        self.profiler.stop("Join")

        if self.is_debug and not self.args.hide_windows and Config.autocam["debug"]["show_split_frames"]:
            for i, bbs_frame in enumerate(bbs_frames):
                self.player.show_frame(bbs_frame, f"bbs_frame {i}")

        return bbs_joined

    def update_camera(self, bbs_joined, frame_orig, frame_orig_debug):
        if not self.is_debug or not self.args.mouse:
            self.profiler.start("Update by BBS")
            self.cameraman.update_camera(bbs_joined)
            self.profiler.stop("Update by BBS")
        elif Config.autocam["debug"]["mouse_use_pid"]:
            self.cameraman._try_update_camera(self.player.mouse_pos)

        # if self.is_debug and self.args.mouse:
        #     self.camera.draw_center_(frame_orig_debug)

        self.profiler.start("Get ROI")
        frame_roi = self.camera.get_frame_roi(frame_orig)
        self.profiler.stop("Get ROI")

        return frame_roi

    def draw_debug_info(self, frame_orig_debug, bbs_joined):
        if not self.is_debug:
            return None

        if not self.args.mouse and Config.autocam["debug"]["draw_detections"]:
            self.detector.draw_bbs_(frame_orig_debug, bbs_joined)
            self.cameraman.draw_ball_prediction_(
                frame_orig_debug, Color.RED)
            self.cameraman.draw_ball_u_(frame_orig_debug, Color.ORANGE)
            self.cameraman.ball_filter.draw_particles_(frame_orig_debug)

        if Config.autocam["debug"]["draw_players_bb"]:
            self.cameraman.draw_players_bb_(frame_orig_debug, bbs_joined)

        frame_roi_debug = self.camera.get_frame_roi(frame_orig_debug)

        if Config.autocam["debug"]["draw_roi"]:
            self.camera.draw_roi_(frame_orig_debug)

        if Config.autocam["debug"]["draw_frame_splitter_roi"]:
            self.frame_splitter.draw_roi_(frame_orig_debug)

        if Config.autocam["debug"]["draw_grid"]:
            self.camera.draw_grid_(frame_orig_debug)

        return frame_roi_debug

    def show_original(self, frame_orig, frame_orig_debug):
        if self.args.hide_windows or not Config.autocam["show_original"]:
            return

        frame = frame_orig_debug if self.is_debug else frame_orig
        self.player.show_frame(frame, "Original")

    def show_top_down(self, bbs_joined):
        draw_players_center = self.is_debug and \
            Config.autocam["debug"]["draw_top_down_players_center"]
        players_center = self.cameraman.players_filter.pos if draw_players_center else None

        frame_top_down = self.top_down.get_frame(
            bbs_joined, players_center
        )

        if not self.args.hide_windows and Config.autocam["show_top_down_window"]:
            self.player.show_frame(frame_top_down, "top down")

        return frame_top_down

    def decorate_roi(self, frame_roi_debug, frame_top_down):
        """Add top down preview and stats bar to the ROI frame."""
        if not self.is_debug or not self.config.autocam["debug"]["decorate_debug_frame"]:
            return frame_roi_debug

        return self.recorder.decorate_frame(frame_roi_debug, frame_top_down)

    def show_roi(self, frame_roi, frame_roi_debug):
        if self.args.hide_windows:
            return

        frame = frame_roi_debug if self.is_debug else frame_roi
        self.player.show_frame(frame, "ROI")

    def write_roi(self, frame_roi, frame_roi_debug):
        if not self.args.record:
            return

        self.recorder.write(frame_roi)

        if self.is_debug:
            self.recorder.write_debug(frame_roi_debug)

    def warp_frame(self, frame_orig):
        frame_orig_masked = self.camera.draw_frame_mask(frame_orig)
        return self.top_down.warp_frame(
            frame_orig_masked,
            overlay=Config.autocam["eval"]["pitch_overlay"]
        )

    def export_for_eval(self, frame_orig_warped, frame_roi, frame_id: int):
        frame_sec = frame_id / int(self.player.fps)
        if not self.args.record or \
                self.args.export_interval_sec == -1 or \
                frame_sec % self.args.export_interval_sec != 0:
            return

        frame_img_id = int(frame_sec // self.args.export_interval_sec)
        self.recorder.save_frame(frame_roi, frame_img_id)
        self.recorder.save_frame(frame_orig_warped, frame_img_id, "warped")

    def show_warped(self, frame_orig_warped):
        if not self.args.hide_windows and self.is_debug and Config.autocam["debug"]["show_warped_frame"]:
            self.player.show_frame(frame_orig_warped, "warped")

    def print_stats(self):
        if Config.autocam["print_profiler_stats"]:
            self.profiler.print_summary()
        if self.is_debug and Config.autocam["debug"]["print_camera_stats"]:
            self.camera.print()

    def process_input(self, key) -> bool:
        if key == ord('m'):
            self.detector_enabled = not self.detector_enabled

        return self.camera.process_input(
            key, self.player.mouse_pos
        )

    def finish(self):
        print(f"Video: {self.config.video_path}")
        self.recorder.release()
        self.player.release()
