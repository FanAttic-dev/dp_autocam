from functools import cached_property
from pathlib import Path

import numpy as np
from main_args import AutocamArgsNamespace
import utils.utils as utils


class Config:
    autocam = utils.load_yaml("./configs/config_autocam.yaml")

    def __init__(self, args: AutocamArgsNamespace):
        self.dataset = Config.load_dataset_config(args)
        self.video_path = Config.get_video_path(self.dataset, args)

    @staticmethod
    def load_dataset_config(args: AutocamArgsNamespace):
        if args.config_path:
            return utils.load_yaml(args.config_path)
        return utils.load_yaml(Config.autocam["dataset"]["config"])

    @staticmethod
    def load_pitch_corners(pts_dict: dict):
        pts = np.array(
            [[v["x"], v["y"]] for v in pts_dict.values()],
            dtype=np.int32
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
