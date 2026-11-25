#!/usr/bin/env bash

set -e

source "$(dirname "${BASH_SOURCE[0]}")/vision_env.sh"

if systemctl is-active --quiet systemd-timesyncd; then
    echo "WARNING: systemd-timesyncd is active."
    echo "Check WSL clock stability before vision tracking."
fi

exec ros2 launch realsense2_camera rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  enable_sync:=true \
  align_depth.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30 \
  enable_infra1:=false \
  enable_infra2:=false \
  pointcloud.enable:=false \
  diagnostics_period:=1.0
