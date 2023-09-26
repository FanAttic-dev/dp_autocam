from functools import cached_property
from pathlib import Path
from utils import coords2pts, load_json
import random


class Config:
    params = {
        "use_trnava_zilina": True,
        "correct_rotation": True,
        "debug": False,
        "verbose": True,
        "eval": {
            "export_enabled": True,
            "export_every_x_seconds": 20,
            "pitch_overlay": True
        },
        "drawing": {
            "enabled": False,
            "show_split_frames": False,
            "show_original": True,
            "show_top_down_window": False,
        },
        "dead_zone": {
            "enabled": False,
            "size": [80, 300]
        },
        "detector": {
            "ball_confidence": 0.35,
            "ball_max_det": 1,
            "players_confidence": 0.25,
        },
        "ball_pf": {
            "dt": 0.1,
            "N": 500,
            "std_pos": 10,
            "std_meas": 150,
            "players_ball_alpha": 0.6
        },
        "players_kf": {
            "dt": 0.1,
            "std_acc": 1,
            "std_meas": 5
        },
        "zoom": {
            "var_min": 100,
            "var_max": 5000
        },
        "u_control": {
            "center": {
                "alpha": 0.1,
                "var_th": 5000
            },
            "velocity": {
                "alpha": 3
            }
        }
    }

    def __init__(self, args):
        self.init_paths(args)
        self.json = load_json(self.config_path)

    def init_paths(self, args):
        if Config.params["use_trnava_zilina"]:
            self.videos_dir = Path("../../datasets/TrnavaZilina/main")
            self.config_path = self.videos_dir / "config.json"
            self.video_path = self.videos_dir / "main_p0_clips/main_p0_clip_04.mp4"
            # self.video_path = self.videos_dir / "TZ_00_22_40__00_24_15.mp4"
        else:
            self.videos_dir = Path(
                "../../datasets/SoccerTrack/wide_view/videos")
            self.config_path = self.videos_dir / "../config.json"
            self.video_path = self.get_random_file(self.videos_dir)
            # self.video_path = self.videos_dir / "F_20200220_1_0120_0150.mp4"

        if args.video_path:
            self.video_path = Path(args.video_path)

        if args.config_path:
            self.config_path = Path(args.config_path)

    @cached_property
    def period(self):
        return "p0" if "p0" in self.video_path.stem else "p1"

    @cached_property
    def pitch_coords(self):
        return self.json["pitch_coords"][self.period]

    @cached_property
    def pitch_coords_pts(self):
        return coords2pts(self.pitch_coords)

    def get_random_file(dir):
        files = list(dir.iterdir())
        idx = random.randint(0, len(files)-1)
        return files[idx]
