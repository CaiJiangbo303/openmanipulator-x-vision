#!/usr/bin/env bash
cd ~/openmanipulator-x-vision
source /opt/ros/humble/setup.bash
source .venv-sam2/bin/activate
source vision_ws/install/setup.bash

ros2 run object_segmenter object_segmenter \
  --ros-args \
  -p use_sim_time:=true

exec bash
