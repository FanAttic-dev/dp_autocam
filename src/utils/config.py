from functools import cached_property
from pathlib import Path

import numpy as np
from utils.argparse import AutocamArgsNamespace
from utils.constants import DT_INT, VIDEO_SUFFIX
import utils.utils as utils


class Config:
    config_autocam_path = "./configs/config_autocam.yaml"
    autocam = utils.load_yaml(config_autocam_path)

    def __init__(self, args: AutocamArgsNamespace):
        self.args = args
        self.dataset = utils.load_yaml(Config.autocam["dataset"]["config"])
        self.video_path = Config.get_video_path(self.dataset, args)

        if args.record:
            self.save_autocam_config()

    @cached_property
    def output_dir(self) -> Path:
        return Path(Config.autocam["recording"]["rec_folder"]) / \
            Config.autocam["recording"]["name"] / \
            self.args.output_sub_dir

    @cached_property
    def output_file_path(self) -> Path:
        self.output_dir.mkdir(exist_ok=True, parents=True)

        video_path = self.video_path.stem
        file_path = self.output_dir / video_path
        file_path = file_path.with_suffix(VIDEO_SUFFIX)

        idx = 1
        while file_path.exists():
            file_path = file_path.with_stem(f"{video_path}_{idx:02}")
            idx += 1

        return file_path

    def save_autocam_config(self):
        file_path = self.output_dir / \
            f"{self.output_file_path.stem}_config_autocam.yaml"
        file_path.parent.mkdir(parents=True, exist_ok=True)
        utils.save_yaml(file_path, Config.autocam)

    @staticmethod
    def load_pitch_corners(pts_dict: dict):
        pts = np.array(
            [[v["x"], v["y"]] for v in pts_dict.values()],
            dtype=DT_INT
        )
        return pts.reshape((-1, 1, 2))

    @staticmethod
    def get_video_path(config, args: AutocamArgsNamespace):
        videos_dir = Path(config["path"])
        video_name = args.video_name if args.video_name else Config.autocam["dataset"]["video"]
        try:
            return next(videos_dir.glob(f"**/{video_name}"))
        except:
            raise FileNotFoundError(f"File {video_name} not found.")

    @cached_property
    def period(self):
        return "p0" if "p0" in self.video_path.stem else "p1"

    @cached_property
    def pitch_corners(self):
        return Config.load_pitch_corners(self.dataset["pitch_corners"][self.period])
