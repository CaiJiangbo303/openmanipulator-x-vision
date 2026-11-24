#!/usr/bin/env bash

set -e

ROOT="$HOME/openmanipulator-x-vision"
WS="$ROOT/vision_ws"
LOG="$ROOT/logs"
PIDFILE="$ROOT/.stack_pids"

MODE="${1:-}"

usage() {
    echo "Usage:"
    echo "  ./start_mode.sh sim"
    echo "  ./start_mode.sh webcam"
    echo "  ./start_mode.sh realsense"
    echo "  ./start_mode.sh stop"
}

if [ -z "$MODE" ]; then
    usage
    exit 1
fi

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

prepare_mode() {
    mkdir -p "$LOG"

    echo "Cleaning previous mode..."
    "$ROOT/clear_all.sh"

    rm -f "$PIDFILE"
}

case "$MODE" in

    sim)
        echo "======================================"
        echo " MODE: SIMULATION"
        echo "======================================"

        # Keep the already validated simulation
        # startup path completely unchanged.
        exec "$ROOT/restart_all.sh"
        ;;


    webcam)
        echo "======================================"
        echo " MODE: WEBCAM RGB DEVELOPMENT"
        echo "======================================"

        prepare_mode

        WIN_HOST=$(
            ip route |
            awk '/default/ {print $3; exit}'
        )

        if [ -z "$WIN_HOST" ]; then
            echo "ERROR: cannot determine Windows host IP."
            exit 1
        fi

        echo
        echo "Windows webcam server must already be running."
        echo "Windows host: $WIN_HOST"
        echo

        start_node "webcam_01_frontend" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash
exec python3 \
  src/webcam_frontend/webcam_frontend/webcam_frontend_node.py \
  --ros-args \
  -p host:='$WIN_HOST' \
  -p port:=5001
"

        sleep 1

        start_node "webcam_02_detector" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 run object_detector object_detector \
  --ros-args \
  --params-file src/object_detector/config/webcam_detector.yaml \
  -p use_sim_time:=false
"

        sleep 1

        start_node "webcam_03_tracker" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash
exec ros2 run object_tracker object_tracker \
  --ros-args \
  -p use_sim_time:=false
"

        sleep 1

        start_node "webcam_04_sam2" "
cd '$ROOT'
source /opt/ros/humble/setup.bash
source .venv-sam2/bin/activate
source vision_ws/install/setup.bash
exec ros2 run object_segmenter object_segmenter \
  --ros-args \
  -p use_sim_time:=false
"

        echo
        echo "WEBCAM perception started."
        echo "RGB only: YOLO + Tracker + SAM2"
        echo "No fake depth/localizer is started."
        ;;


    realsense)
        echo "======================================"
        echo " MODE: REALSENSE RGB-D"
        echo "======================================"

        prepare_mode

        source /opt/ros/humble/setup.bash
        source "$WS/install/setup.bash"

        if ! ros2 pkg prefix realsense2_camera \
            >/dev/null 2>&1; then

            echo "ERROR: ROS2 package realsense2_camera not found."
            exit 1
        fi

        start_node "realsense_01_camera" "
source /opt/ros/humble/setup.bash
source '$WS/install/setup.bash'

exec ros2 launch \
  realsense2_camera \
  rs_launch.py \
  enable_color:=true \
  enable_depth:=true \
  enable_sync:=true \
  align_depth.enable:=true \
  rgb_camera.color_profile:=640x480x30 \
  depth_module.depth_profile:=640x480x30 \
  wait_for_device_timeout:=10.0
"

        echo
        echo "Checking RealSense topics..."

        READY=0

        for i in {1..12}; do
            if ros2 topic list 2>/dev/null |
                grep -qx \
                '/camera/camera/color/image_raw'
            then
                READY=1
                break
            fi

            sleep 1
        done

        if [ "$READY" -ne 1 ]; then
            echo
            echo "ERROR: RealSense color stream was not found."
            echo "Check:"
            echo "  $LOG/realsense_01_camera.log"
            exit 1
        fi

        echo "RealSense color stream detected."

        start_node "realsense_02_detector" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash

exec ros2 run object_detector object_detector \
  --ros-args \
  --params-file src/object_detector/config/detector.yaml \
  -p backend:=yolo \
  -p image_topic:=/camera/camera/color/image_raw \
  -p use_sim_time:=false
"

        sleep 1

        start_node "realsense_03_tracker" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash

exec ros2 run object_tracker object_tracker \
  --ros-args \
  -p use_sim_time:=false
"

        sleep 1

        start_node "realsense_04_sam2" "
cd '$ROOT'
source /opt/ros/humble/setup.bash
source .venv-sam2/bin/activate
source vision_ws/install/setup.bash

exec python -m object_segmenter.segmenter_node \
  --ros-args \
  -p use_sim_time:=false \
  -p image_topic:=/camera/camera/color/image_raw
"

        sleep 1

        # BENCH MODE:
        # Until the real vehicle/world pose source is connected,
        # localize in the RealSense color optical frame.
        #
        # Aligned depth is geometrically registered to color.
        start_node "realsense_05_localizer" "
cd '$WS'
source /opt/ros/humble/setup.bash
source install/setup.bash

exec ros2 run object_localizer mask_localizer \
  --ros-args \
  --params-file src/object_localizer/config/realsense.yaml \
  -p use_sim_time:=false \
  -p world_frame:=camera_color_optical_frame
"

        echo
        echo "======================================"
        echo " REALSENSE PERCEPTION STARTED"
        echo "======================================"
        echo
        echo "Current hardware stage:"
        echo "  RealSense RGB"
        echo "      -> YOLO11s"
        echo "      -> Tracker"
        echo "      -> SAM2"
        echo "  Aligned depth + color CameraInfo"
        echo "      -> mask_localizer"
        echo "      -> camera_color_optical_frame XYZ"
        echo
        echo "Self Filter / Feasibility / Selector are"
        echo "intentionally NOT started in RealSense"
        echo "bench mode until real vehicle/arm TF exists."
        ;;


    stop)
        exec "$ROOT/clear_all.sh"
        ;;


    *)
        echo "ERROR: unknown mode '$MODE'"
        usage
        exit 1
        ;;

esac
