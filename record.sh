VIDEOS_PATH='../../datasets/TrnavaZilina/videos/first_half_clips'
CONFIG_PATH=$VIDEOS_PATH/../../config.json
for clip in $VIDEOS_PATH/*.mp4 
do
    python main.py --video-path $clip --config-path $CONFIG_PATH --record --hide-windows
done
