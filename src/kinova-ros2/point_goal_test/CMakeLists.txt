cmake_minimum_required(VERSION 3.14)
project(point_goal_test)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# ---- build-tool dependencies ----
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

# ---- C++ runtime dependencies ----
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(trajectory_msgs REQUIRED)

# ---- C++ executable ----
add_executable(feeding_safe_mover src/feeding_safe_mover.cpp)
ament_target_dependencies(feeding_safe_mover
  rclcpp
  geometry_msgs
  moveit_ros_planning_interface
  moveit_msgs
  trajectory_msgs
)
install(TARGETS feeding_safe_mover
  DESTINATION lib/${PROJECT_NAME})

# ---- Python package (makes `import point_goal_test` work for the scripts) ----
ament_python_install_package(${PROJECT_NAME})

# Install the Python scripts as ros2-run-able executables.
# Each install() call renames the .py file to the entry-point name that
# existed in the original setup.py console_scripts, preserving ros2 run
# compatibility: `ros2 run point_goal_test point_selector_node`
install(PROGRAMS point_goal_test/point_selector_node.py
  RENAME point_selector_node
  DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS point_goal_test/arm_mover_node.py
  RENAME arm_mover_node
  DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS point_goal_test/test_scoop_gazebo.py
  RENAME test_scoop_gazebo
  DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS point_goal_test/gazebo_teleop.py
  RENAME gazebo_teleop
  DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS point_goal_test/mouth_goal_node.py
  RENAME mouth_goal_node
  DESTINATION lib/${PROJECT_NAME})

# ---- shared resources ----
install(DIRECTORY launch config worlds
  DESTINATION share/${PROJECT_NAME})
install(FILES resource/${PROJECT_NAME}
  DESTINATION share/ament_index/resource_index/packages)
install(FILES package.xml
  DESTINATION share/${PROJECT_NAME})

ament_package()
