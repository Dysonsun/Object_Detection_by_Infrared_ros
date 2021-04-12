cmake_minimum_required(VERSION 2.8.12)

project(object_msgs)

set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")

find_package(catkin REQUIRED
        COMPONENTS
        actionlib_msgs
        geometry_msgs
        sensor_msgs
        std_msgs
        cv_bridge
        message_generation
        )

add_message_files(
        FILES
        BoundingBox.msg
        BoundingBoxes.msg
        ObjectCoordinate.msg
        ObjectCoordinates.msg
)

generate_messages(
        DEPENDENCIES
        geometry_msgs
        sensor_msgs
        std_msgs
)

catkin_package(
        CATKIN_DEPENDS
        geometry_msgs
        sensor_msgs
        cv_bridge
        message_runtime
        std_msgs
)
