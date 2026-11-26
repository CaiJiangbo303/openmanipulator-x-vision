#!/usr/bin/env bash
set -eo pipefail
ws=$(realpath "$1")
test -d "$ws/vision_ws/src"
cd "$ws"
exec env -i HOME="$HOME" PATH=/usr/bin:/bin LANG=C.UTF-8 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 CMAKE_BUILD_PARALLEL_LEVEL=3 MAKEFLAGS=-j3 bash --noprofile --norc -c '
set -eo pipefail
source /opt/ros/humble/setup.bash
env | sort > build_environment.txt
timeout -k 10 600 colcon --log-base log build --base-paths vision_ws/src --build-base build --install-base install --parallel-workers 2 --cmake-args -DBUILD_TESTING=OFF
'
