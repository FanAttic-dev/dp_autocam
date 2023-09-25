#!/bin/bash
VAR_FOLDER='../../datasets/TrnavaZilina/VAR'

for clip_folder in var_p0_clips var_p1_clips
do
	for clip in $VAR_FOLDER/$clip_folder/*.mp4
	do
		clip_frames_folder="${clip%.*}_frames"
		frame_prefix=`basename "${clip%.*}"`
		mkdir $clip_frames_folder
		ffmpeg -i $clip -r 1/20 -vsync 1 -s 1280x720 $clip_frames_folder/${frame_prefix}_%04d.jpg
	done
done