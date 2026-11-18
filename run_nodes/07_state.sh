#!/usr/bin/env bash
cd ~/openmanipulator-x-vision/vision_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run object_state_manager object_state_manager \
  --ros-args \
  -p use_sim_time:=true \
  -p measurement_topic:=/object_self_filter/detections_3d

exec bash
