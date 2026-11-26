#!/usr/bin/env bash
# Invoked by the runner in an empty environment. No user shell profiles.
set -eo pipefail
if [[ "$MATRIX" == C ]]; then
    source "$SITE_SETUP"
else
    source /opt/ros/humble/setup.bash
    source "$SOURCE_ROOT/install/local_setup.bash"
    export PROD_NODES="$SOURCE_ROOT/run_nodes"
    export PYTHONPATH="$PROD_NODES:$KIN_PARENT:$HARNESS:${PYTHONPATH:-}"
fi
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID="$TEST_DOMAIN" RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOG_DIR="$RUN/roslogs" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
mkdir -p "$ROS_LOG_DIR"
exec "$@"
