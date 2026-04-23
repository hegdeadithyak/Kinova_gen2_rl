cmake_minimum_required(VERSION 3.5)
project(click_pointer)

find_package(ament_cmake REQUIRED)

install(PROGRAMS scripts/click_pointer.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME click_pointer
)

ament_package()
