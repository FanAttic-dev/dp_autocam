
import argparse


class AutocamArgsNamespace(argparse.Namespace):
    record: bool
    output_sub_dir: str
    mouse: bool
    video_name: str
    hide_windows: bool
    export_frames: bool
    no_debug: bool


def parse_args():
    parser = argparse.ArgumentParser()
    ns = AutocamArgsNamespace()
    parser.add_argument('-r', "--record", action='store_true',
                        help="Export output as video.")
    parser.add_argument("--output-sub-dir", action='store', required=False, default="",
                        help="Output sub-directory under rec_path defined by config_autocam.")
    parser.add_argument('-m', "--mouse", action='store_true',
                        help="Debug mode for moving the camera with mouse.")
    parser.add_argument('-v', "--video-name", action='store', required=False)
    parser.add_argument("--hide-windows", action='store_true', default=False,
                        help="Hide all windows while running.")
    parser.add_argument("--export-frames", action='store_true', default=False,
                        help="Export frames every X seconds (used for evaluation).")
    parser.add_argument("--no-debug", action='store_true', default=False,
                        help="Do not generate debug frame.")
    return parser.parse_args(namespace=ns)
