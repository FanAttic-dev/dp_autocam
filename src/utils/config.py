from functools import cached_property
from pathlib import Path
from utils.helpers import coords2pts, load_json
import random


class Config:
    autocam = load_json("./configs/config_autocam.json")

    def __init__(self, args):
        self.dataset = Config.load_dataset_config(args)
        self.video_path = Config.get_video_path(self.dataset, args)

    @staticmethod
    def load_dataset_config(args):
        if args.config_path:
            return load_json(args.config_path)
        return load_json(Config.autocam["dataset"]["config"])

    @staticmethod
    def get_video_path(json, args):
        videos_dir = Path(json["path"])
        video_path = args.video_path if args.video_path else Config.autocam["dataset"]["video"]
        return videos_dir / video_path

    @cached_property
    def period(self):
        return "p0" if "p0" in self.video_path.stem else "p1"

    @cached_property
    def pitch_coords(self):
        return self.dataset["pitch_coords"][self.period]

    @cached_property
    def pitch_coords_pts(self):
        return coords2pts(self.pitch_coords)

    def get_random_file(dir):
        files = list(dir.iterdir())
        idx = random.randint(0, len(files)-1)
        return files[idx]