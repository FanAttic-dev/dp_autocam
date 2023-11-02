# Autocam
Autocam is an automatic virtual cameraman. The input is a wide-angle recording of a soccer match containing the full pitch. Autocam automatically finds the interesting region of each frame, crops it so that it emulates pan-tilt-zoom of a camera on a tripod, and finally exports it as a new video.

## Getting started

1. Create a [Conda environment](https://docs.conda.io/en/latest/) by installing, e.g., [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html) and then running these commands:

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
  - Contains, e.g., camera PTZ (pan-tilt-zoom) default values and limits, or coordinates of pitch corners.
- App config: `configs/config_autocam.yaml`
  - Contains, e.g., paths for dataset and output, constants for debugging, detection, filters, PID, or zoom limits.

For more details, refer to the comments next to the parameters.

## Scripts

Under `/scripts`, you can find shell scripts for recording full matches (`record_full_match.sh`) or folders of video clips (`record_clips.sh`).

## TODO

- [ ] Try running it on Windows
- [ ] Check if chmod +x for scripts necessary.
