from pathlib import Path
import cv2

from image_processor import process_frame


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


img_path = Path("/home/atti/source/datasets/videos/sample_wide.mp4")

cap = cv2.VideoCapture(str(img_path))

ret, frame = get_next_frame(cap)
if ret:
    process_frame(frame)

cap.release()
cv2.destroyAllWindows()
