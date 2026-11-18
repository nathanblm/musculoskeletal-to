cmake_minimum_required(VERSION 3.0.2)
project(fatigue_control_ros)

find_package(catkin REQUIRED COMPONENTS cv_bridge fatigue_control_msgs rospy sensor_msgs)
catkin_python_setup()
catkin_package(CATKIN_DEPENDS cv_bridge fatigue_control_msgs rospy sensor_msgs)

catkin_install_python(PROGRAMS
  scripts/human_state_estimator
  scripts/model_viewer
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  catkin_install_python(PROGRAMS test/test_ros1.py
    DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
  add_rostest(test/ros1.test)
endif()
