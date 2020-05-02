#!/usr/bin/env bash
A=0
DURATION="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 sample.mp4)"
DURATION=${DURATION%.*}
CLIPLENGTH=10
while [ $A -le $DURATION ]
do
	ffmpeg -ss $A -i sample.mp4 -t $CLIPLENGTH -c:v libx264 output-${A}.mp4;
	let "A=A+$CLIPLENGTH"
done
