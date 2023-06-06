from pathlib import Path
import cv2
import json

from image_processor import BackgroundSubtractor, process_frame


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


videos_path = Path("/home/atti/source/datasets/videos/")
video_name = Path("sample_wide.mp4")

cap = cv2.VideoCapture(str(videos_path / video_name))
with open(videos_path / f"coords_{video_name.stem}.json", 'r') as f:
    coords = json.load(f)

bgSubtractor = BackgroundSubtractor()
# bgSubtractor.init(cap)

while True:
    # ret, frame = get_frame_at(cap, 27)
    ret, frame = get_next_frame(cap)
    if not ret:
        break

    proceed = process_frame(frame, coords, bgSubtractor)
    if not proceed:
        break

cap.release()
cv2.destroyAllWindows()
