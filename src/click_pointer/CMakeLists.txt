cmake_minimum_required(VERSION 3.5)
project(click_pointer)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/click_pointer.py
  scripts/fork_acquisition.py
  scripts/click_to_point.py
  DESTINATION lib/${PROJECT_NAME}
)

install(FILES
  scripts/tui.py
  scripts/face_overlay.py
  scripts/algo.py
  scripts/utils.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
