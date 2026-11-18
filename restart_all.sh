#!/usr/bin/env bash

set -e

ROOT="$HOME/openmanipulator-x-vision"
WS="$ROOT/vision_ws"
LOG="$ROOT/logs"
PIDFILE="$ROOT/.stack_pids"

mkdir -p "$LOG"
rm -f "$PIDFILE"

echo "======================================"
echo " OpenManipulator-X FULL RESTART"
echo "======================================"

echo
echo "[1/3] Cleaning old stack..."

"$ROOT/clear_all.sh"

echo
echo "Starting ROS 2 daemon..."

source /opt/ros/humble/setup.bash
ros2 daemon start >/dev/null 2>&1 || true

sleep 2

start_node() {
    NAME="$1"
    CMD="$2"

    echo "Starting $NAME ..."

    nohup setsid bash -lc "$CMD" \
        > "$LOG/$NAME.log" \
        2>&1 &

    LEADER_PID=$!

    sleep 0.2

    PGID=$(
        ps -o pgid= -p "$LEADER_PID" 2>/dev/null \
        | tr -d ' '
    )

    if [ -z "$PGID" ]; then
        echo "ERROR: failed to obtain PGID for $NAME"
        return 1
    fi

    echo "$PGID $NAME" >> "$PIDFILE"

    echo "  PGID=$PGID"
}

echo
echo "[2/3] Starting simulation..."

start_node "01_sim" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 launch vision_sim vision_sim.launch.py
"

sleep 7

echo
echo "[3/3] Starting stack..."

start_node "02_detector" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 run object_detector object_detector \
  --ros-args \
  --params-file src/object_detector/config/detector.yaml \
  -p backend:=color_debug \
  -p use_sim_time:=true
"

sleep 1

start_node "03_tracker" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 run object_tracker object_tracker \
  --ros-args \
  -p use_sim_time:=true
"

sleep 1

start_node "04_sam2" "
cd '$ROOT'
source /opt/ros/humble/setup.bash
source .venv-sam2/bin/activate
source vision_ws/install/setup.bash
exec ros2 run object_segmenter object_segmenter \
  --ros-args \
  -p use_sim_time:=true
"

sleep 1

start_node "05_localizer" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 run object_localizer mask_localizer \
  --ros-args \
  -p use_sim_time:=true
"

sleep 1

start_node "06_self_filter" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 run object_self_filter object_self_filter \
  --ros-args \
  -p use_sim_time:=true
"

sleep 1

start_node "07_state_manager" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 run object_state_manager object_state_manager \
  --ros-args \
  -p use_sim_time:=true \
  -p measurement_topic:=/object_self_filter/detections_3d
"

sleep 1

start_node "08_predictor" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 run object_motion_predictor object_motion_predictor \
  --ros-args \
  -p use_sim_time:=true
"

sleep 1

start_node "09_feasibility" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 run four_dof_feasibility four_dof_feasibility \
  --ros-args \
  -p use_sim_time:=true
"

sleep 1

start_node "10_selector" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 run target_selector target_selector \
  --ros-args \
  -p use_sim_time:=true
"

sleep 1

start_node "11_grasp_bridge" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 run grasp_command_bridge grasp_command_bridge \
  --ros-args \
  -p use_sim_time:=true \
  -p dry_run:=true
"

echo
echo "======================================"
echo " FULL STACK STARTED"
echo "======================================"
echo
echo "Process groups:"
cat "$PIDFILE"
echo
echo "Grasp Bridge remains DRY-RUN ONLY."
