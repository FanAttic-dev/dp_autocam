from autocam import Autocam
from utils.argparse import parse_args
from utils.config import Config
import utils.utils as utils


if __name__ == "__main__":
    args = parse_args()
    # args.record = True
    # args.mouse = True
    config = Config(args)

    try:
        autocam = Autocam(args, config)
        autocam.run()
        autocam.finish()
    except Exception as e:
        utils.save_txt(
            config.output_dir / f"{config.output_file_path.stem}_err_log.txt",
            str(e))
        raise e
