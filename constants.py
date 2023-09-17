from pathlib import Path

from utils import get_random_file

params = {
    "use_trnava_zilina": True,
    "correct_rotation": False,
    "show_split_frames": True,
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
            "alpha": 2
        }
    }
}


if params["use_trnava_zilina"]:
    videos_dir = Path("../../datasets/TrnavaZilina/videos")
    config_path = videos_dir / "../config.json"
    # video_path = videos_dir / "clip01.mp4"
    # video_path = videos_dir / "TZ_00_22_40__00_24_15.mp4"
    # video_path = videos_dir / "first_half.mp4"
    video_path = videos_dir / "clips" / "clip_first_half_00.mp4"

else:
    videos_dir = Path(
        "../../datasets/SoccerTrack/wide_view/videos")
    config_path = videos_dir / "../config.json"
    video_path = get_random_file(videos_dir)
    # video_path = videos_dir / "F_20200220_1_0120_0150.mp4"


# BGR
colors = {
    "white": (255, 255, 255),
    "red": (51, 74, 227),
    "green": (95, 162, 44),
    "blue": (202, 162, 67),
    "violet": (167, 86, 136),
    "orange": (132, 187, 253),
    "teal": (187, 205, 127),
    "yellow": (0.1250*255, 0.6940*255, 0.9290*255)
}
