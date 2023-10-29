#!/bin/bash

DIR_PATH='../../datasets/TrnavaZilina/main/clips'

time {
    for clip_path in $DIR_PATH/*.mp4
    do
        video_name=${clip_path##*/}
        python src/main.py --video-name $video_name --record --output-sub-dir clips --export-interval-sec 1 --hide-windows
    done
}