#!/usr/bin/env bash
cd ~/openmanipulator-x-vision/vision_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run object_self_filter object_self_filter \
  --ros-args \
  -p use_sim_time:=true

exec bash
