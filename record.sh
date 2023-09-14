VIDEOS_PATH='../../datasets/TrnavaZilina/videos/clips'
CONFIG_PATH=$VIDEOS_PATH/../../config.json
for clip in $VIDEOS_PATH/*.mp4 
do
    nice -n 20 python index.py --video-path $clip --config-path $CONFIG_PATH --record --hide-windows
done
