# RealSense Runtime Stability Baseline

Date: 2026-11-25

## Environment

- Ubuntu 22.04 / WSL2
- ROS 2 Humble
- WSL kernel: 6.18.58.1
- RealSense D435
- Firmware: 5.17.0.10
- Librealsense: 2.58.4
- RealSense ROS driver: 4.58.4
- Cyclone DDS: 0.10.5

## Observed Problems

1. WSL wall-clock discontinuity of approximately 1179.7 ms.
2. Severe intermittent RGB-D image delivery gaps.
3. Metadata frequency did not always reflect image delivery.
4. BEST_EFFORT camera publishing was incompatible with
   the existing RELIABLE localizer subscription.

## Runtime Mitigations

- Use rmw_cyclonedds_cpp.
- Use ROS_DOMAIN_ID=197 for isolated bench testing.
- Use ROS_LOCALHOST_ONLY=1.
- Preserve default camera QoS for localizer compatibility.
- Keep RGB-D synchronization and depth alignment enabled.
- Temporarily stop systemd-timesyncd if needed during testing.
- No firmware, SDK or kernel upgrade performed.

## DDS Comparison

Both comparison runs used SENSOR_DATA camera QoS.

Metric                       Fast DDS    Cyclone DDS
RGB image rate (Hz)           17.16       29.98
Aligned depth rate (Hz)        5.53       29.98
Max depth receive gap (ms)  1133.19       34.75
Missing depth counters       172          0

Four subsequent 15-second Cyclone DDS image tests:
- Minimum depth rate: 29.85 Hz
- Maximum depth receive gap: 67.45 ms
- Gaps above 100 ms: 0

## Restored Pipeline

Sequential measurements after runtime recovery:

- YOLOE: 29.91 Hz
- Object Tracker: 29.98 Hz
- Fast 3D: 29.99 Hz
- Mask 3D: 29.97 Hz
- State Manager: 30.00 Hz

A valid fused blue-block state was observed:
- Confidence: 0.5387
- Measurement age: 3.44 ms
- Source age: approximately 69.19 ms
- valid_3d: true
- stale: false
- missed: 0

## Usage

Source vision_env.sh in every vision terminal.

Launch the camera with launch_camera.sh only when no
other process is using the RealSense device.

Temporary clock isolation:
    sudo systemctl stop systemd-timesyncd

Restore time synchronization after bench testing:
    sudo systemctl start systemd-timesyncd

Build the C++ diagnostic:
    cmake -S cpp_probe -B /tmp/d435_cpp_probe_build \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build /tmp/d435_cpp_probe_build -j2

## Remaining Validation

- Long-duration clock stability.
- Default QoS stability under full perception workload.
- Dynamic target identity and detection continuity.
- Unique valid 3D measurement update rate.
- Tracking authorization and physical robot safety.

This baseline does not establish the exact internal
Fast DDS failure mechanism or prove that intermittent
USB faults have been eliminated.

Published message rate is not equivalent to valid
measurement rate.

No physical robot motion was validated in this session.
