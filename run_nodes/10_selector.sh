#!/usr/bin/env bash
cd ~/openmanipulator-x-vision/vision_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run target_selector target_selector \
  --ros-args \
  -p use_sim_time:=true

exec bash
