VIDEOS_PATH='../../datasets/TrnavaZilina/videos/clips'
CONFIG_PATH=$VIDEOS_PATH/../../config.json
for clip in $VIDEOS_PATH/*.mp4 
do
    python index.py --video_path $clip --config_path $CONFIG_PATH --record
done
