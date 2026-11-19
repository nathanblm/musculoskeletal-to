cmake_minimum_required(VERSION 3.0.2)
project(human_musculoskeletal)

find_package(catkin REQUIRED COMPONENTS cv_bridge message_generation roscpp rospy sensor_msgs std_msgs)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GAZEBO REQUIRED gazebo)
catkin_python_setup()
add_message_files(FILES MuscleState.msg)
generate_messages(DEPENDENCIES std_msgs)
catkin_package(CATKIN_DEPENDS cv_bridge message_runtime roscpp rospy sensor_msgs std_msgs)

link_directories(${GAZEBO_LIBRARY_DIRS})
add_executable(gazebo_muscle_state_bridge src/gazebo_muscle_state_bridge.cpp)
target_include_directories(gazebo_muscle_state_bridge PRIVATE ${catkin_INCLUDE_DIRS} ${GAZEBO_INCLUDE_DIRS})
target_compile_options(gazebo_muscle_state_bridge PRIVATE ${GAZEBO_CFLAGS_OTHER})
target_link_libraries(gazebo_muscle_state_bridge ${catkin_LIBRARIES} ${GAZEBO_LIBRARIES})
add_dependencies(gazebo_muscle_state_bridge ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
install(TARGETS gazebo_muscle_state_bridge RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

catkin_install_python(PROGRAMS
  scripts/human_state_publisher
  scripts/gazebo_joint_state_bridge
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch description data DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
