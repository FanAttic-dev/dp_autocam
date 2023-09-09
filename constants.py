from pathlib import Path


# videos_dir = Path("/home/atti/source/datasets/SoccerTrack/wide_view/videos")
# coords_path = videos_dir / "../../coords.json"

videos_dir = Path("/home/atti/source/datasets/TrnavaZilina/videos")
coords_path = videos_dir / "../coords.json"


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
