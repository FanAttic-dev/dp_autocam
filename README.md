# Autocam

Autocam is an automatic virtual cameraman. The input is a wide-angle rectilinear video of a soccer match containing the full pitch. Autocam automatically finds the interesting region of each frame, crops it so that it emulates the pan-tilt-zoom (PTZ) of a camera on a tripod, and finally exports it as a new video.

The system was developed as a part of a [diploma thesis](https://is.muni.cz/th/l78dh/?lang=en) by Attila Zsíros in 2023.

![Title main](assets/images/title_main.png)

![Title ROI](assets/images/title_roi.png)

## Getting started

1. Create a [Conda environment](https://docs.conda.io/en/latest/) by first installing, e.g., [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) and then running these commands:

    ```shell
    conda env create -f environment.yml
    conda activate autocam
    ```

2. Run Autocam on a provided sample video using the Makefile:

    ```shell
    # Run with debug visualizations.
    make debug

    # Run and control the camera by WASD keys.
    # Use the `mouse_use_pid` option in config_autocam.yaml to control the camera by the mouse.
    make debug_mouse

    # Run without visualizations.
    make no_debug

    # Record the whole clip into the "./recordings folder".
    make record
    ```

3. Use the `spacebar` to proceed to the next frame and `q` to exit.

## User control

- `spacebar` — proceed to the next frame;
- `q` — exit;
- `-` and `+` — zoom (focal length);
- `7` and `9` — pitch tilt correction angle;
- `4` and `6` — original lens field of view;
- `2` and `8` — sensor size;
- `WASD` — camera movement (only in mouse mode, i.e., when executed by `make debug_mouse`);

## Config files

Under the `/configs` folder, there are two types of config files:

- Dataset config: `config_trnava_zilina.yaml`
  - Contains, e.g., the path to the dataset, camera PTZ (pan-tilt-zoom) default values and limits, or coordinates of pitch corners.
- App config: `configs/config_autocam.yaml`
  - Contains, e.g., paths for dataset and output, constants for debugging, detection, filters, PID, or zoom limits.

For more details, refer to the parameter's descriptions in the config files.

## Scripts

Under `/scripts`, you can find shell scripts for:

- Recording full matches (`record_full_match.sh`).
  - It assumes that in the dataset folder (specified in the dataset config), there are two files:
    - `main_p0.mp4` for the first period of the match
    - `main_p1.mp4` for the second period of the match
- Recording folders of video clips (`record_clips.sh`).
  - It records all `*.mp4` files located in the specified folder.

## Methods

For a detailed description of the methods, refer to the [diploma thesis](https://is.muni.cz/th/l78dh/?lang=en).

|              Detection Pipeline              |          Virtual Camera Algorithm          |
| :------------------------------------------: | :----------------------------------------: |
| ![](assets/images/Detection_pipeline@2x.png) | ![](assets/images/virtual_camera_algo.png) |

