from pathlib import Path

from utils import get_random_file

constants = {
    "rotate_cameras": False,
    "use_trnava_zilina": True
}


if constants["use_trnava_zilina"]:
    videos_dir = Path("/home/atti/source/datasets/TrnavaZilina/videos")
    config_path = videos_dir / "../config.json"
    # video_path = videos_dir / "clip01.mp4"
    video_path = videos_dir / "TZ_00_22_40__00_24_15.mp4"
else:
    videos_dir = Path(
        "/home/atti/source/datasets/SoccerTrack/wide_view/videos")
    config_path = videos_dir / "../../config.json"
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
