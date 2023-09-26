#!/bin/bash
VIDEOS_PATH='../../datasets/TrnavaZilina/main'
CONFIG_PATH=$VIDEOS_PATH/config.json
for period in main_p0.mp4 main_p1.mp4
do
    python main.py --video-path $VIDEOS_PATH/$period --config-path $CONFIG_PATH --record --hide-windows
done
