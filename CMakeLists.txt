cmake_minimum_required(VERSION 3.5)

project(avm_app)

# OpenCV
find_package(OpenCV REQUIRED)

# ROS 2
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)


include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ./srcs
)

message(STATUS "opencv inc path " ${OpenCV_INCLUDE_DIRS})

add_executable(avm_cali avm_cali_demo.cpp ./srcs/common.cpp)
add_executable(avm_app avm_app_demo.cpp ./srcs/common.cpp)
add_executable(surround_view_live surround_view_live.cpp ./srcs/common.cpp)
add_executable(surround_view_ros surround_view_ros.cpp ./srcs/common.cpp)

target_link_libraries(avm_cali ${OpenCV_LIBS})
target_link_libraries(avm_app ${OpenCV_LIBS})
target_link_libraries(surround_view_live ${OpenCV_LIBS})
target_link_libraries(surround_view_ros
    ${OpenCV_LIBS}
    rclcpp::rclcpp
)

ament_target_dependencies(surround_view_ros
  sensor_msgs
  cv_bridge
)