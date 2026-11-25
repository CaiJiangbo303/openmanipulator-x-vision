#!/usr/bin/env bash

VISION_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

source /opt/ros/humble/setup.bash
source "$VISION_ROOT/vision_ws/install/setup.bash"

export ROS_DOMAIN_ID=197
export ROS_LOCALHOST_ONLY=1
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

echo "Vision environment ready: Cyclone DDS, Domain 197"
