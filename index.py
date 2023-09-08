from pathlib import Path
import cv2
from camera import PerspectiveCamera
from constants import videos_dir, coords_path
from detector import YoloBallDetector, YoloPlayerDetector
from frame_splitter import PerspectiveFrameSplitter
from utils import add_bb_, add_bb_ball_, get_bounding_box, get_random_file
from top_down import TopDown
from utils import load_json
from video_player import VideoPlayer
from video_recorder import VideoRecorder
import argparse
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

video_path = get_random_file(videos_dir)
video_path = Path(
    "/home/atti/source/datasets/SoccerTrack/wide_view/videos/F_20200220_1_0120_0150.mp4")
player = VideoPlayer(video_path)
delay = player.get_delay(args.record)

is_alive, frame_orig = player.get_next_frame()
camera = PerspectiveCamera(frame_orig)
frame_splitter = PerspectiveFrameSplitter(frame_orig)

pitch_coords = load_json(coords_path)
top_down = TopDown(pitch_coords, camera)
detector = YoloPlayerDetector(pitch_coords)
ball_detector = YoloBallDetector(pitch_coords, camera.ball_model)

# args.record = True

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
    frame_orig = detector.preprocess(frame_orig)

    """ Detection """
    bbs_joined = {
        "boxes": [],
        "cls": []
    }
    bbs_ball_joined = {
        "boxes": [],
        "cls": []
    }
    if not camera.pause_measurements and not args.mouse:
        """ Split frame, detect objects, merge & draw bounding boxes """
        frames = frame_splitter.split(frame_orig)

        # Players
        bbs, _ = detector.detect(frames)
        bbs_joined = frame_splitter.join_bbs(bbs)
        detector.draw_bbs_(frame_orig, bbs_joined)

        # Balls
        bbs_ball, bbs_ball_frame = ball_detector.detect(frames)
        # for i, ball_frame in enumerate(bbs_ball_frame):
        #     player.show_frame(ball_frame, f"ball frame {i}")
        bbs_ball_joined = frame_splitter.join_bbs(bbs_ball)
        detector.draw_bbs_(frame_orig, bbs_ball_joined, colors["white"])

    """ ROI """
    if args.mouse:
        camera.model.set_target(mousePos["x"])
        camera.model.update()
        _, center_y = camera.center
        center_x = camera.model.get()
        camera.set_center(center_x, center_y)

        camera.draw_center_(frame_orig)
    else:
        camera.update_by_bbs(bbs_joined, bbs_ball_joined, top_down)
        camera.draw_ball_prediction_(frame_orig, colors["green"])
        camera.ball_model.draw_particles_(frame_orig)

    frame = camera.get_frame(frame_orig)

    # camera.draw_dead_zone_(frame)
    # player.show_frame(frame, "ROI")
    # camera.print()
    # frame_splitter.draw_roi_(frame_orig)
    # camera.draw_roi_(frame_orig)

    # x1, y1, x2, y2 = get_bounding_box(bbs_joined)
    # cv2.rectangle(frame_orig, (x1, y1), (x2, y2),
    #               colors["green"], thickness=10)

    player.show_frame(frame_orig, "Original")

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
