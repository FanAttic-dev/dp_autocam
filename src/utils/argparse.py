
import argparse


class AutocamArgsNamespace(argparse.Namespace):
    record: bool
    output_sub_dir: str
    export_interval_sec: int
    mouse: bool
    video_name: str
    hide_windows: bool
    no_debug: bool
    ignore_bounds: bool


def parse_args():
    parser = argparse.ArgumentParser()
    ns = AutocamArgsNamespace()
    parser.add_argument('-r', "--record", action='store_true',
                        help="Export output as video.")
    parser.add_argument("--output-sub-dir", action='store', required=False, default="",
                        help="Output sub-directory under rec_path defined by config_autocam.")
    parser.add_argument("--export-interval-sec", action='store', type=int, required=False, default=-1,
                        help="Time interval to export a warped frame [sec].")
    parser.add_argument('-m', "--mouse", action='store_true',
                        help="Debug mode for moving the camera with mouse.")
    parser.add_argument('-v', "--video-name", action='store', required=False)
    parser.add_argument("--hide-windows", action='store_true', default=False,
                        help="Hide all windows while running.")
    parser.add_argument("--no-debug", action='store_true', default=False,
                        help="Do not generate debug frame.")
    parser.add_argument("--ignore-bounds", action='store_true', default=False,
                        help="Ignore camera PTZ bounds. May result in sampling outside of the original image.")
    return parser.parse_args(namespace=ns)
