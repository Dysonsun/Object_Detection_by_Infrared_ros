#!/usr/bin/env zsh


DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
echo ${DIR}

source ./devel/setup.zsh --extend
source ~/software/catkin_workspace/install/setup.zsh --extend
condaenv
source activate yolov3_i
cd scripts
python detect_and_publish_ros.py --data ./data/custom.data --cfg ./cfg/yolov3-spp-r.cfg --weights ./weights/final.pt --img_topic /thermal/image_raw/compressed --publish
