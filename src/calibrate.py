from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import utils
from utils.config import Config
from evaluation.tvcalib_wrapper import TVCalibWrapper


dataset_config = utils.load_yaml(Config.autocam["dataset"]["config"])
images_path = Path(dataset_config["reference_path"]) / "full"
output_dir = Path("tmp")

tvcalib_wrapper = TVCalibWrapper(images_path, output_dir)

# Segment
image_ids, keypoints_raw = tvcalib_wrapper.segment()

# Calibrate
df = tvcalib_wrapper.calibrate(image_ids, keypoints_raw)

# Warp & save
for idx, sample in df.iterrows():
    img_warped = tvcalib_wrapper.warp_frame(sample, overlay=True)
    tvcalib_wrapper.save_frame(sample, img_warped)

df_losses = pd.DataFrame(df, columns=[
    "homography", "loss_ndc_lines", "loss_ndc_circles", "loss_ndc_total"
])
df_losses.to_csv(images_path / "losses.csv")
