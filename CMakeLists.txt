cmake_minimum_required(VERSION 2.8.3)
project(fast_mask_segmentation)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)

add_message_files(
  FILES
  BoundingBox2D.msg
  FastMaskBB2D.msg
  FastMaskBB2DArray.msg
)

add_service_files(
  FILES
  FastMaskSegmentation.srv
)

generate_messages(
  DEPENDENCIES
  std_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS
  rospy
)
