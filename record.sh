#!/bin/bash
VIDEOS_PATH='../../datasets/TrnavaZilina/videos'
CONFIG_PATH=$VIDEOS_PATH/../config.json
for clip in $VIDEOS_PATH/main_p0_clips/*.mp4 $VIDEOS_PATH/main_p1_clips/*.mp4
do
    python main.py --video-path $clip --config-path $CONFIG_PATH --record --hide-windows
done
