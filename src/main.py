from autocam import Autocam
from utils.argparse import parse_args
from utils.config import Config
import utils.utils as utils


if __name__ == "__main__":
    args = parse_args()
    # args.record = True
    args.mouse = True

    config = Config(args)
    autocam = Autocam(args, config)

    try:
        autocam.run()
    except Exception as e:
        err_log_path = config.output_dir / \
            f"{config.output_file_path.stem}_err_log.txt"
        utils.save_txt(err_log_path, str(e))
        raise e
