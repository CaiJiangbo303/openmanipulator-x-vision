cmake_minimum_required(VERSION 3.8)
project(d435_probe)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(probe probe.cpp)
ament_target_dependencies(probe rclcpp sensor_msgs)
