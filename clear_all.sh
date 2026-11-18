#!/usr/bin/env bash

set +e

ROOT="$HOME/openmanipulator-x-vision"
PIDFILE="$ROOT/.stack_pids"

echo "======================================"
echo " Cleaning OpenManipulator-X stack"
echo "======================================"

echo "[1/5] Stopping recorded process groups..."

if [ -f "$PIDFILE" ]; then

    while read -r PGID NAME; do
        [ -z "$PGID" ] && continue
        echo "Stopping $NAME (PGID $PGID)"
        kill -INT -- "-$PGID" 2>/dev/null
    done < "$PIDFILE"

    sleep 2

    while read -r PGID NAME; do
        [ -z "$PGID" ] && continue
        kill -TERM -- "-$PGID" 2>/dev/null
    done < "$PIDFILE"

    sleep 2

    while read -r PGID NAME; do
        [ -z "$PGID" ] && continue
        kill -KILL -- "-$PGID" 2>/dev/null
    done < "$PIDFILE"
fi

rm -f "$PIDFILE"


echo "[2/5] Cleaning orphan project nodes..."

PATTERNS=(
    "/rgbd_frontend/"
    "/object_detector/"
    "/object_tracker/"
    "/object_segmenter/"
    "ros2 run object_segmenter object_segmenter"
    "object_segmenter.segmenter_node"
    "/object_localizer/"
    "/object_self_filter/"
    "/object_state_manager/"
    "/object_motion_predictor/"
    "/four_dof_feasibility/"
    "/target_selector/"
    "/grasp_command_bridge/"
    "/webcam_frontend/"
    "webcam_frontend_node.py"
    "/vision_debug_viewer/"
    "vision_sim.launch.py"
    "simulation_arm_tf.py"
    "realsense2_camera"
    "rs_launch.py"
    "simulation_tf.py"
    "__node:=vision_robot_state_publisher"
)

for pattern in "${PATTERNS[@]}"; do
    pkill -TERM -f "$pattern" 2>/dev/null
done

sleep 2

# Hard-kill anything that ignored TERM.
for pattern in "${PATTERNS[@]}"; do
    pkill -KILL -f "$pattern" 2>/dev/null
done


echo "[3/5] Cleaning ROS-Gazebo bridge leftovers..."

pkill -TERM -f "ros_gz_bridge" 2>/dev/null
sleep 1
pkill -KILL -f "ros_gz_bridge" 2>/dev/null


echo "[4/5] Killing stale Gazebo instances..."

pkill -TERM -f "ign gazebo" 2>/dev/null
pkill -TERM -f "gz sim" 2>/dev/null
pkill -TERM -f "ignition-gazebo" 2>/dev/null

sleep 2

pkill -KILL -f "ign gazebo" 2>/dev/null
pkill -KILL -f "gz sim" 2>/dev/null
pkill -KILL -f "ignition-gazebo" 2>/dev/null


echo "[5/5] Resetting ROS daemon..."

source /opt/ros/humble/setup.bash

ros2 daemon stop >/dev/null 2>&1

sleep 1


echo
echo "===== Remaining Gazebo ====="

pgrep -af \
'ign gazebo|gz sim|ignition-gazebo' \
|| echo "NONE"


echo
echo "===== Remaining project processes ====="

pgrep -af \
'rgbd_frontend|object_detector|object_tracker|object_segmenter|mask_localizer|object_self_filter|object_state_manager|object_motion_predictor|four_dof_feasibility|target_selector|grasp_command_bridge|simulation_tf.py|simulation_arm_tf.py|vision_sim.launch.py' \
|| echo "NONE"


echo
echo "======================================"
echo " CLEAN COMPLETE"
echo "======================================"
