import cv2
from camera import PerspectiveCamera
from constants import PAN_DX, TILT_DY, ZOOM_DZ, videos_dir, coords_path
from detector import YoloPlayerDetector
from frame_splitter import PerspectiveFrameSplitter
from utils import get_random_file
from top_down import TopDown
from utils import load_json
from video_player import VideoPlayer

video_name = get_random_file(videos_dir)
print(f"Video: {video_name}")
player = VideoPlayer(video_name)

pitch_coords = load_json(coords_path)
top_down = TopDown(pitch_coords)
detector = YoloPlayerDetector(pitch_coords)

_, frame_orig = player.get_next_frame()
camera = PerspectiveCamera(frame_orig)
frame_splitter = PerspectiveFrameSplitter(frame_orig)

i = 0
while True:
    ret, frame_orig = player.get_next_frame()
    if not ret:
        break

    h, w, _ = frame_orig.shape
    frame_orig = detector.preprocess(frame_orig)

    # Split, detect & merge
    frames = frame_splitter.split(frame_orig)
    bbs, _ = detector.detect(frames)
    frame_splitter.draw_roi_(frame_orig)

    bbs_joined = frame_splitter.join_bbs(bbs)
    detector.draw_bounding_boxes_(frame_orig, bbs_joined)

    # frame_warped = top_down.warp_frame(frame_orig)
    # player.show_frame(frame_warped, "warped")

    # ROI
    # camera.update_by_bbs(bbs)
    frame = camera.get_frame(frame_orig)
    player.show_frame(frame, "ROI")
    camera.print()

    # Top-down
    bbs_pts = top_down.warp_bbs(bbs_joined)
    top_down_frame = top_down.pitch_model.copy()
    top_down.draw_roi_(top_down_frame, camera)
    top_down.draw_points_(top_down_frame, bbs_pts)
    player.show_frame(top_down_frame, "top down")

    camera.draw_roi_(frame_orig)
    player.show_frame(frame_orig, "Original")

    key = cv2.waitKey(0)
    if key == ord('d'):
        camera.pan(PAN_DX)
    elif key == ord('a'):
        camera.pan(-PAN_DX)
    elif key == ord('w'):
        camera.tilt(-TILT_DY)
    elif key == ord('s'):
        camera.tilt(TILT_DY)
    elif key == ord('p'):
        camera.zoom(ZOOM_DZ)
    elif key == ord('m'):
        camera.zoom(-ZOOM_DZ)
    elif key == ord('r'):
        camera.reset()
    elif key == ord('q'):
        break

    i += 1

print(f"Video: {video_name}")
player.release()
