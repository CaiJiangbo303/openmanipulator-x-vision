#!/usr/bin/env bash
cd ~/openmanipulator-x-vision/vision_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

ros2 run four_dof_feasibility four_dof_feasibility \
  --ros-args \
  -p use_sim_time:=true

exec bash
