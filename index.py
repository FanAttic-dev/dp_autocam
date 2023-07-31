import cv2
from camera import PerspectiveCamera
from constants import videos_dir, coords_path
from detector import YoloPlayerDetector
from frame_splitter import PerspectiveFrameSplitter
from utils import get_random_file
from top_down import TopDown
from utils import load_json
from video_player import VideoPlayer

""" Init """
video_name = get_random_file(videos_dir)
print(f"Video: {video_name}")
player = VideoPlayer(video_name)

pitch_coords = load_json(coords_path)
top_down = TopDown(pitch_coords)
detector = YoloPlayerDetector(pitch_coords)

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
    bbs, _ = detector.detect(frames)
    frame_splitter.draw_roi_(frame_orig)
    bbs_joined = frame_splitter.join_bbs(bbs)
    detector.draw_bbs_(frame_orig, bbs_joined)

    """ ROI """
    # camera.update_by_bbs(bbs)
    frame = camera.get_frame(frame_orig)
    player.show_frame(frame, "ROI")
    camera.print()
    camera.draw_roi_(frame_orig)
    player.show_frame(frame_orig, "Original")

    """ Top-down """
    bbs_pts = top_down.warp_bbs(bbs_joined)
    top_down_frame = top_down.pitch_model.copy()
    top_down.draw_roi_(top_down_frame, camera)
    top_down.draw_points_(top_down_frame, bbs_pts)
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
