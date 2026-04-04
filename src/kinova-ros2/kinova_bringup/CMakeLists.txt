cmake_minimum_required(VERSION 3.8)
project(kinova_bringup)

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Eigen3 REQUIRED)

# ── Planner plugin shared library ─────────────────────────────────────────────
# Builds: lib/libelliptic_planner_plugin.so
# This is what pluginlib loads at runtime when MoveIt2 looks up
# "elliptic_planner/EllipticPlannerManager"
add_library(elliptic_planner_plugin SHARED
  src/elliptic_planner_plugin.cpp
)

target_include_directories(elliptic_planner_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${EIGEN3_INCLUDE_DIRS}
)

ament_target_dependencies(elliptic_planner_plugin
  rclcpp
  moveit_core
  moveit_ros_planning
  pluginlib
)

# Required: register the plugin XML with pluginlib
pluginlib_export_plugin_description_file(
  moveit_core
  elliptic_planner_plugin_description.xml
)
install(PROGRAMS scripts/pick_laddoo_node.py DESTINATION lib/${PROJECT_NAME})
# ── Python scripts ────────────────────────────────────────────────────────────
install(PROGRAMS
  scripts/elliptic_delivery.py
  scripts/feeding_node.py
  DESTINATION lib/${PROJECT_NAME}
)

# ── Install plugin library ────────────────────────────────────────────────────
install(TARGETS elliptic_planner_plugin
  EXPORT export_${PROJECT_NAME}
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
)

install(FILES elliptic_planner_plugin_description.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

# ── Install launch / config ───────────────────────────────────────────────────
install(DIRECTORY launch        DESTINATION share/${PROJECT_NAME})
install(DIRECTORY moveit_resource DESTINATION share/${PROJECT_NAME})
install(DIRECTORY worlds/       DESTINATION share/${PROJECT_NAME}/worlds)

ament_package()