#!/bin/bash
CONFIG_PATH='./configs/config_trnava_zilina.yaml'
DIR_PATH='../../datasets/TrnavaZilina/main/clips'
for clip_path in $DIR_PATH/*.mp4
do
    video_name=${clip_path##*/}
    python src/main.py --video-name $video_name --config-path $CONFIG_PATH --record --hide-windows --export-frames
done
