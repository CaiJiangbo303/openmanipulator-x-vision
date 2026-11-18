#!/usr/bin/env bash
cd ~/openmanipulator-x-vision/vision_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run object_detector object_detector \
  --ros-args \
  --params-file src/object_detector/config/detector.yaml \
  -p backend:=color_debug \
  -p use_sim_time:=true

exec bash
