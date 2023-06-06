from pathlib import Path
import cv2
import json
from camera import FixedHeightCamera
from constants import WINDOW_FLAGS, WINDOW_NAME

from image_processor import ImageProcessor


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


videos_path = Path("/home/atti/source/datasets/videos/")
video_name = Path("sample_wide.mp4")

cap = cv2.VideoCapture(str(videos_path / video_name))
with open(videos_path / f"coords_{video_name.stem}.json", 'r') as f:
    coords = json.load(f)

img_processor = ImageProcessor()
ret, frame = get_next_frame(cap)
camera = FixedHeightCamera(frame)

while True:
    ret, frame = get_next_frame(cap)
    if not ret:
        break

    show_frame(frame, window_name=f"{WINDOW_NAME} original")

    frame = img_processor.process_frame(frame, coords)
    frame = camera.get_frame(frame)
    camera.pan(-10)
    if frame is not None:
        show_frame(frame)

    key = cv2.waitKey(0)
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
