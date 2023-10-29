from functools import cached_property
from pathlib import Path

import numpy as np
from utils.argparse import AutocamArgsNamespace
from utils.constants import DT_INT
import utils.utils as utils


class Config:
    config_autocam_path = "./configs/config_autocam.yaml"
    autocam = utils.load_yaml(config_autocam_path)

    def __init__(self, args: AutocamArgsNamespace):
        self.dataset = utils.load_yaml(Config.autocam["dataset"]["config"])
        self.video_path = Config.get_video_path(self.dataset, args)
        self.output_dir = \
            Path(Config.autocam["recording"]["rec_folder"]) / \
            Config.autocam["recording"]["name"] / \
            args.output_sub_dir

        if args.record:
            self.save_autocam_config()

    def save_autocam_config(self):
        file_path = self.output_dir / "config_autocam.yaml"
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
