import cv2


class VideoPlayer:
    WINDOW_NAME = 'frame'
    WINDOW_FLAGS = cv2.WINDOW_NORMAL  # cv2.WINDOW_AUTOSIZE

    def __init__(self, video_name):
        self.cap = cv2.VideoCapture(str(video_name.absolute()))

    def get_frame_at(self, seconds):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        frame_id = int(fps*seconds)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        return self.get_next_frame(self.cap)

    def get_next_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            print('Could not receive frame.')
        return ret, frame

    def play_video(self):
        while True:
            ret, frame = self.get_next_frame(self.cap)
            if not ret:
                break
            cv2.imshow('frame', frame)
            key = cv2.waitKey(0)
            if key == ord('q'):
                break

    def create_window(self, window_name):
        cv2.namedWindow(window_name, VideoPlayer.WINDOW_FLAGS)

    def show_frame(self, frame, window_name=WINDOW_NAME):
        self.create_window(window_name)
        cv2.imshow(window_name, frame)

    def release(self):
        self.cap.release()
        cv2.destroyAllWindows()
