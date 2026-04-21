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
install(PROGRAMS scripts/kinova_trajectory_bridge.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME kinova_trajectory_bridge
)
install(PROGRAMS scripts/joint_state_relay.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME joint_state_relay
)
install(PROGRAMS scripts/scoop_action.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME scoop_action
)

ament_package()