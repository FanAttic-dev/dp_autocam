#!/bin/bash
CONFIG_PATH='./configs/config_trnava_zilina.yaml'
for clip in main_p0.mp4 main_p1.mp4
do
    python src/main.py --video-name $clip --config-path $CONFIG_PATH --record --hide-windows
done
