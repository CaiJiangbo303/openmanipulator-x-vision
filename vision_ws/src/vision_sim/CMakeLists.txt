cmake_minimum_required(VERSION 3.8)

project(vision_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    worlds
    launch
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/simulation_tf.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
