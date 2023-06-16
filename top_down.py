import json
from pathlib import Path
import cv2
from utils import coords_to_pts

from constants import WINDOW_NAME


class TopDown:
    assets_path = Path('assets')
    pitch_model_path = assets_path / 'pitch_model.png'
    pitch_coords_path = assets_path / 'coords_pitch_model.json'

    def __init__(self, video_pitch_coords):
        self.pitch_model = cv2.imread(str(TopDown.pitch_model_path))
        with open(TopDown.pitch_coords_path, 'r') as f:
            self.pitch_coords = json.load(f)
        self.H, _ = cv2.findHomography(coords_to_pts(video_pitch_coords),
                                       coords_to_pts(self.pitch_coords))

    def warp_frame(self, frame):
        return cv2.warpPerspective(
            frame, self.H, (self.pitch_model.shape[1], self.pitch_model.shape[0]))
