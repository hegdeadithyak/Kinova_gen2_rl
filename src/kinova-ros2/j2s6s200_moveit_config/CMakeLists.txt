cmake_minimum_required(VERSION 3.8)
project(j2s6s200_moveit_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch urdf
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
