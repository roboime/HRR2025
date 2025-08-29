cmake_minimum_required(VERSION 3.8)
project(motion)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

add_executable(imu_driver src/imu_driver.cpp)
ament_target_dependencies(imu_driver rclcpp sensor_msgs std_msgs)

install(TARGETS imu_driver
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  src/walking_controller.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()


