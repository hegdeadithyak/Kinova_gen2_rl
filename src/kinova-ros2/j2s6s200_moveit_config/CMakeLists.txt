cmake_minimum_required(VERSION 3.8)
project(j2s6s200_moveit_config)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch urdf
  DESTINATION share/${PROJECT_NAME}
)
install(PROGRAMS scripts/kinova_sdk_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME kinova_sdk_node
)
install(PROGRAMS scripts/trajectory_executor.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME trajectory_executor
)
install(PROGRAMS
  scripts/kinova_sdk_node.py
  scripts/trajectory_executor.py
  scripts/kinova_trajectory_bridge.py
  scripts/joint_state_relay.py
  DESTINATION lib/${PROJECT_NAME}
)

# Add inside CMakeLists.txt (before ament_package())
install(PROGRAMS
  scripts/joint_state_relay.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME joint_state_relay
)
ament_package()
