from collections import defaultdict
import random
from argparse import Namespace
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Union
import cv2

import numpy as np
import kornia
import torch
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
from SoccerNet.Evaluation.utils_calibration import SoccerPitch

from tvcalib.cam_modules import SNProjectiveCamera
from tvcalib.module import TVCalibModule
from tvcalib.cam_distr.tv_main_tribune import get_cam_distr, get_dist_distr
from sn_segmentation.src.custom_extremities import generate_class_synthesis, get_line_extremities
from tvcalib.sncalib_dataset import custom_list_collate, split_circle_central
from tvcalib.utils.io import detach_dict, tensor2list
from tvcalib.utils.objects_3d import SoccerPitchLineCircleSegments, SoccerPitchSNCircleCentralSplit
from tvcalib.inference import InferenceDatasetCalibration, InferenceDatasetSegmentation, InferenceSegmentationModel
from tvcalib.inference import get_camera_from_per_sample_output
from tvcalib.utils import visualization_mpl_min as viz
from utils.config import Config


class TVCalibWrapper:
    def __init__(self, images_path, output_dir):
        self.args = Namespace(
            images_path=images_path,
            output_dir=output_dir,
            checkpoint="assets/weights/segment_localization/train_59.pt",
            gpu=True,
            nworkers=1,
            batch_size_seg=16,
            batch_size_calib=256,
            image_width=1280,
            image_height=720,
            optim_steps=5000,
            lens_dist=False,
            write_masks=False
        )
        self.device = "cuda" if self.args.gpu and torch.cuda.is_available() else "cpu"
        self.object3d = SoccerPitchLineCircleSegments(
            device=self.device, base_field=SoccerPitchSNCircleCentralSplit()
        )
        self.object3dcpu = SoccerPitchLineCircleSegments(
            device="cpu", base_field=SoccerPitchSNCircleCentralSplit()
        )

        sn_pitch = SoccerPitch()
        corners_pitch = Config.pitch_corners

        corners_pitch_norm = np.array([
            sn_pitch.bottom_left_corner,
            sn_pitch.top_left_corner,
            sn_pitch.top_right_corner,
            sn_pitch.bottom_right_corner,
        ])

        self.H_norm, _ = cv2.findHomography(corners_pitch_norm, corners_pitch)

        self.pitch_model = cv2.imread("./assets/pitch_model.png")
        self.pitch_h, self.pitch_w, _ = self.pitch_model.shape
        self.pitch_model_red = self.get_pitch_model_red()

    def get_pitch_model_red(self):
        pitch = self.pitch_model
        pitch[:, :, 0] = 0
        pitch[:, :, 1] = 0
        return pitch

    def segment(self):
        print("Segmentation start")

        lines_palette = [0, 0, 0]
        for line_class in SoccerPitch.lines_classes:
            lines_palette.extend(SoccerPitch.palette[line_class])

        fn_generate_class_synthesis = partial(
            generate_class_synthesis, radius=4)
        fn_get_line_extremities = partial(get_line_extremities, maxdist=30,
                                          width=455, height=256, num_points_lines=4, num_points_circles=8)

        model_seg = InferenceSegmentationModel(
            self.args.checkpoint, self.device)
        dataset_seg = InferenceDatasetSegmentation(
            self.args.images_path, self.args.image_width, self.args.image_height
        )
        print("number of images:", len(dataset_seg))
        dataloader_seg = torch.utils.data.DataLoader(
            dataset_seg,
            batch_size=self.args.batch_size_seg,
            num_workers=self.args.nworkers,
            shuffle=False,
            collate_fn=custom_list_collate,
        )

        image_ids = []
        keypoints_raw = []
        (self.args.output_dir / "masks").mkdir(parents=True, exist_ok=True)
        for batch_dict in tqdm(dataloader_seg):
            # semantic segmentation
            # image_raw: [B, 3, image_height, image_width]
            # image: [B, 3, 256, 455]
            with torch.no_grad():
                sem_lines = model_seg.inference(
                    batch_dict["image"].to(self.device))
            sem_lines = sem_lines.cpu().numpy().astype(
                np.uint8)  # [B, 256, 455]

            # point selection
            with Pool(self.args.nworkers) as p:
                skeletons_batch = p.map(fn_generate_class_synthesis, sem_lines)
                keypoints_raw_batch = p.map(
                    fn_get_line_extremities, skeletons_batch)

            # write to file
            if self.args.write_masks:
                print("Write masks to file")
                for image_id, mask in zip(batch_dict["image_id"], sem_lines):
                    mask = Image.fromarray(mask.astype(np.uint8)).convert("P")
                    mask.putpalette(lines_palette)
                    mask.convert("RGB").save(
                        self.args.output_dir / "masks" / Path(image_id).name)

            image_ids.extend(batch_dict["image_id"])
            keypoints_raw.extend(keypoints_raw_batch)

        print("Segmentation end")
        return image_ids, keypoints_raw

    def calibrate(self, image_ids, keypoints_raw):
        print("Calibration start")

        model_calib = TVCalibModule(
            self.object3d,
            get_cam_distr(1.96, self.args.batch_size_calib, 1),
            get_dist_distr(self.args.batch_size_calib,
                           1) if self.args.lens_dist else None,
            (self.args.image_height, self.args.image_width),
            self.args.optim_steps,
            self.device,
            log_per_step=True,
            tqdm_kwqargs=None,
        )

        dataset_calib = InferenceDatasetCalibration(
            keypoints_raw, self.args.image_width, self.args.image_height, self.object3d)
        dataloader_calib = torch.utils.data.DataLoader(
            dataset_calib, self.args.batch_size_calib, collate_fn=custom_list_collate)

        per_sample_output = defaultdict(list)
        per_sample_output["image_id"] = [[x] for x in image_ids]
        for x_dict in dataloader_calib:
            _batch_size = x_dict["lines__ndc_projected_selection_shuffled"].shape[0]

            points_line = x_dict["lines__px_projected_selection_shuffled"]
            points_circle = x_dict["circles__px_projected_selection_shuffled"]
            print(f"{points_line.shape=}, {points_circle.shape=}")

            per_sample_loss, cam, _ = model_calib.self_optim_batch(x_dict)
            output_dict = tensor2list(detach_dict(
                {**cam.get_parameters(_batch_size), **per_sample_loss}))

            output_dict["points_line"] = points_line
            output_dict["points_circle"] = points_circle
            for k in output_dict.keys():
                per_sample_output[k].extend(output_dict[k])

        df = pd.DataFrame.from_dict(per_sample_output)

        df = df.explode(
            column=[k for k, v in per_sample_output.items() if isinstance(v, list)])
        df.set_index("image_id", inplace=True, drop=False)

        print("Calibration end")
        return df

    def visualize_per_sample_output(self, df, save=False):
        for i in range(len(df)):
            sample = df.iloc[i]

            image_id = Path(sample.image_id).stem
            print(f"[{image_id}] loss: {sample.loss_ndc_total}")
            image = Image.open(self.args.images_path /
                               sample.image_id).convert("RGB")
            image = T.functional.to_tensor(image)

            cam = get_camera_from_per_sample_output(
                sample, self.args.lens_dist)
            # print(cam, cam.str_lens_distortion_coeff(
            #     b=0) if self.args.lens_dist else "")
            points_line, points_circle = sample["points_line"], sample["points_circle"]

            if self.args.lens_dist:
                # we visualize annotated points and image after undistortion
                image = cam.undistort_images(
                    image.unsqueeze(0).unsqueeze(0)).squeeze()
                # print(points_line.shape) # expected: (1, 1, 3, S, N)
                points_line = SNProjectiveCamera.static_undistort_points(
                    points_line.unsqueeze(0).unsqueeze(0), cam).squeeze()
                points_circle = SNProjectiveCamera.static_undistort_points(
                    points_circle.unsqueeze(0).unsqueeze(0), cam).squeeze()
            else:
                psi = None

            fig, ax = viz.init_figure(
                self.args.image_width, self.args.image_height)
            ax = viz.draw_image(ax, image)
            ax = viz.draw_reprojection(ax, self.object3dcpu, cam)
            ax = viz.draw_selected_points(
                ax,
                self.object3dcpu,
                points_line,
                points_circle,
                kwargs_outer={
                    "zorder": 1000,
                    "rasterized": False,
                    "s": 500,
                    "alpha": 0.3,
                    "facecolor": "none",
                    "linewidths": 3,
                },
                kwargs_inner={
                    "zorder": 1000,
                    "rasterized": False,
                    "s": 50,
                    "marker": ".",
                    "color": "k",
                    "linewidths": 4.0,
                },
            )
            plt.show()

            if save:
                dpi = 50
                # plt.savefig(self.args.output_dir / f"{image_id}.pdf", dpi=dpi)
                # plt.savefig(self.args.output_dir / f"{image_id}.svg", dpi=dpi)
                plt.savefig(self.args.output_dir / f"{image_id}.png", dpi=dpi)
                sample.to_csv(self.args.output_dir / f"{image_id}.csv")

    def init_figure(self):
        figsize = (16, 9)
        fig, ax = plt.subplots(figsize=figsize)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        return fig, ax

    def show_img(self, img, ax=None):
        if ax is None:
            _, ax = self.init_figure()

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        plt.show()

    def img_id2img_path(self, image_id):
        return self.args.images_path / image_id

    def warp_frame(self, sample, overlay=False):
        H_frame = np.array(sample["homography"])
        H = self.H_norm @ H_frame

        img_path = self.img_id2img_path(sample.image_id)
        img = cv2.imread(str(img_path))
        img_warped = cv2.warpPerspective(img, H, (self.pitch_w, self.pitch_h))

        if overlay:
            img_warped = cv2.add(
                img_warped, (self.pitch_model_red * 0.5).astype(np.uint8))

        return img_warped

    def save_frame(self, sample, img_warped):
        img_path = self.img_id2img_path(sample.image_id)
        stem = img_path.stem + "_warped"
        img_warped_path = img_path.with_stem(stem)
        cv2.imwrite(str(img_warped_path), img_warped)
