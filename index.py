from pathlib import Path
import cv2
import json
from camera import FixedHeightCamera
from constants import PAN_DX, WINDOW_FLAGS, WINDOW_NAME
import random
from detector import BgDetector, YoloPlayerDetector

from top_down import TopDown
from utils import split_frame


def get_frame_at(cap, seconds):
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_id = int(fps*seconds)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    return get_next_frame(cap)


def get_next_frame(cap):
    ret, frame = cap.read()
    if not ret:
        print('Could not receive frame.')
    return ret, frame


def play_video(cap):
    while True:
        ret, frame = get_next_frame(cap)
        if not ret:
            break
        cv2.imshow('frame', frame)
        key = cv2.waitKey(0)
        if key == ord('q'):
            break


def show_frame(frame, window_name=WINDOW_NAME):
    cv2.namedWindow(window_name, WINDOW_FLAGS)
    cv2.imshow(window_name, frame)


def get_random_file(dir):
    files = list(dir.iterdir())
    idx = random.randint(0, len(files)-1)
    return files[idx]


videos_dir = Path("/home/atti/source/datasets/SoccerTrack/wide_view/videos")
coords_path = videos_dir / "../../coords.json"
# video_name = get_random_file(videos_dir)
# video_name = videos_dir / "F_20200220_1_0180_0210.mp4"
video_name = videos_dir / "F_20200220_1_0480_0510.mp4"
print(f"Video: {video_name}")

cap = cv2.VideoCapture(str(video_name.absolute()))
with open(coords_path, 'r') as f:
    pitch_coords = json.load(f)

ret, frame = get_next_frame(cap)
camera = FixedHeightCamera(frame)
top_down = TopDown(pitch_coords)
detector = YoloPlayerDetector(pitch_coords)

i = 0
while True:
    ret, frame = get_next_frame(cap)
    if not ret:
        break

    h, w, _ = frame.shape

    # frame_warped = top_down.warp_frame(frame)
    # show_frame(frame_warped, "warped")

    # bb_pts = top_down.warp_bbs(bbs)
    # top_down_frame = top_down.draw_points(bb_pts)
    # show_frame(top_down_frame, "top down")

    frame = detector.preprocess(frame)
    # frame = camera.get_frame(frame)
    frames = split_frame(frame)
    for i, frame in enumerate(frames):
        bbs, frame = detector.detect(frame)
        show_frame(frame, f"frame {i}")
    # camera.update_by_bbs(bbs)
    # show_frame(mask, window_name=f"{WINDOW_NAME} mask")

    # if frame is not None:
    #     show_frame(frame)

    key = cv2.waitKey(0)
    if key == ord('d'):
        camera.pan(PAN_DX)
    elif key == ord('a'):
        camera.pan(-PAN_DX)
    elif key == ord('q'):
        break

    i += 1

print(f"Video: {video_name}")
cap.release()
cv2.destroyAllWindows()
