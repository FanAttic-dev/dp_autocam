#!/bin/bash

time {
    for clip in main_p0.mp4 main_p1.mp4
    do
        echo "Running $clip in parallel"
        python src/main.py --video-name $clip --record --output-sub-dir full --hide-windows --export-frames 2> ./recordings/${clip}_err_log.txt &
    done
    wait
}