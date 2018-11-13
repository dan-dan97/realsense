cmake_minimum_required(VERSION 2.8.3)
project(cv3d_hangover)
set(CMAKE_CXX_STANDARD 11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  tf
)

find_package(PCL REQUIRED)
find_library(LIBREALSENSE2_LIB realsense2 REQUIRED)

catkin_package(
	INCLUDE_DIRS include
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_definitions(${PCL_DEFINITIONS})
add_executable(cv3d_hangover src/objects_segmentation.cpp)

target_link_libraries(cv3d_hangover
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${LIBREALSENSE2_LIB}
)
