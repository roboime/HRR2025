cmake_minimum_required(VERSION 3.8)
project(roboime_navigation)

# Set C++ standard
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rclpy REQUIRED)
find_package(roboime_msgs REQUIRED)

# Include directories
include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# =============================================================================
# C++ CORE LIBRARY
# =============================================================================

add_library(navigation_core STATIC
  src/particle_filter.cpp
  src/ekf_localization.cpp
)

ament_target_dependencies(navigation_core
  rclcpp
  std_msgs
  std_srvs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  roboime_msgs
)

target_include_directories(navigation_core PUBLIC include)

# =============================================================================
# C++ EXECUTABLES - Algoritmos de Alta Performance
# =============================================================================

# Particle Filter SLAM Node
add_executable(particle_filter_node
  src/particle_filter_node.cpp
)

target_link_libraries(particle_filter_node navigation_core)

ament_target_dependencies(particle_filter_node
  rclcpp
  std_msgs
  std_srvs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  roboime_msgs
)

# EKF Localization Node
add_executable(ekf_localization_node
  src/ekf_localization_node.cpp
)

target_link_libraries(ekf_localization_node navigation_core)

ament_target_dependencies(ekf_localization_node
  rclcpp
  std_msgs
  std_srvs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  roboime_msgs
)

# Navigation Coordinator Node
add_executable(navigation_coordinator_node
  src/navigation_coordinator_node.cpp
  src/navigation_coordinator.cpp
)

target_link_libraries(navigation_coordinator_node navigation_core)

ament_target_dependencies(navigation_coordinator_node
  rclcpp
  std_msgs
  std_srvs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  roboime_msgs
)

# =============================================================================
# PYTHON MODULES - Algoritmos de Planejamento
# =============================================================================

# Install Python modules
ament_python_install_package(navigation)

# Install Python scripts
install(PROGRAMS
  navigation/localization_node.py
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  navigation/path_planner_node.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME path_planner_node
)

install(PROGRAMS
  navigation/navigation_manager.py
  DESTINATION lib/${PROJECT_NAME}
  RENAME navigation_manager
)

# =============================================================================
# INSTALL TARGETS
# =============================================================================

# Install C++ executables
install(TARGETS
  particle_filter_node
  ekf_localization_node
  navigation_coordinator_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install header files
install(DIRECTORY include/
  DESTINATION include/
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch/
)

# Install config files
install(DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config/
)

# Install resource marker file
install(DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}/resource/
)

# =============================================================================
# TESTING
# =============================================================================

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_particle_filter.cpp")
    ament_add_gtest(test_particle_filter test/test_particle_filter.cpp)
  endif()

  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/test/test_ekf_localization.cpp")
    ament_add_gtest(test_ekf_localization test/test_ekf_localization.cpp)
  endif()
endif()

# =============================================================================
# EXPORT
# =============================================================================

ament_export_include_directories(include)
ament_export_dependencies(
  rclcpp
  std_msgs
  geometry_msgs
  nav_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
  tf2_eigen
  roboime_msgs
)

ament_package() 