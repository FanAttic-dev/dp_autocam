# Autocam

Autocam is an automatic virtual cameraman. The input is a wide-angle recording of a soccer match containing the full pitch. Autocam automatically finds the interesting region of each frame, crops it so that it emulates pan-tilt-zoom of a camera on a tripod, and finally exports it as a new video.

- [ ] Title image

## Getting started

1. Create a [Conda environment](https://docs.conda.io/en/latest/) by first installing, e.g., [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) and then running these commands:

    ```shell
    conda env create -f environment.yml
    conda activate autocam
    ```

2. Run Autocam on a sample video.

    ```shell
    python src/main.py
    ```

    or using the provided Makefile:

    ```shell
    make run
    ```

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

## TODO

- [ ] Try running it on Windows
- [ ] Mention spaces (ROI, Original frame, Top-down)
- [ ] The application allows for visualization of the current state, including the region of interest (ROI) for each camera, detections
      predicted camera center, and states for different modules , such as the PTZ or the PID. The presence of a particular visualization is controlled by the user in the configuration file.
- [ ] Debug mode description
