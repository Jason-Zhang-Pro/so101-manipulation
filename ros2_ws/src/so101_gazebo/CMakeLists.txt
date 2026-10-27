cmake_minimum_required(VERSION 3.8)
project(so101_gazebo)

find_package(ament_cmake REQUIRED)
find_package(so101_description REQUIRED)
find_package(ignition-gazebo6 REQUIRED)
find_package(ignition-msgs8 REQUIRED)
find_package(ignition-plugin1 REQUIRED COMPONENTS register)
find_package(ignition-rendering6 REQUIRED)
find_package(ignition-transport11 REQUIRED)

add_library(so101-environment-randomization-system SHARED
  src/environment_randomization_system.cpp
)

target_compile_features(
  so101-environment-randomization-system
  PRIVATE cxx_std_17
)

target_link_libraries(
  so101-environment-randomization-system
  PRIVATE
    ignition-gazebo6::ignition-gazebo6
    ignition-msgs8::ignition-msgs8
    ignition-plugin1::ignition-plugin1
    ignition-rendering6::ignition-rendering6
    ignition-transport11::ignition-transport11
)

install(
  TARGETS so101-environment-randomization-system
  LIBRARY DESTINATION lib
)

add_library(so101-contact-evidence-system SHARED src/contact_evidence_system.cpp)
target_compile_features(so101-contact-evidence-system PRIVATE cxx_std_17)
target_link_libraries(so101-contact-evidence-system PRIVATE
  ignition-gazebo6::ignition-gazebo6
  ignition-msgs8::ignition-msgs8
  ignition-plugin1::ignition-plugin1
  ignition-transport11::ignition-transport11)
install(TARGETS so101-contact-evidence-system LIBRARY DESTINATION lib)

add_library(so101-grasp-attachment-system SHARED src/grasp_attachment_system.cpp)
target_compile_features(so101-grasp-attachment-system PRIVATE cxx_std_17)
target_link_libraries(so101-grasp-attachment-system PRIVATE
  ignition-gazebo6::ignition-gazebo6
  ignition-msgs8::ignition-msgs8
  ignition-plugin1::ignition-plugin1
  ignition-transport11::ignition-transport11)
install(TARGETS so101-grasp-attachment-system LIBRARY DESTINATION lib)

if(BUILD_TESTING)
  enable_testing()
  add_executable(contact-evidence-snapshot-test tests/contact_evidence_snapshot_test.cpp)
  target_compile_features(contact-evidence-snapshot-test PRIVATE cxx_std_17)
  target_link_libraries(contact-evidence-snapshot-test PRIVATE
    ignition-gazebo6::ignition-gazebo6 ignition-msgs8::ignition-msgs8)
  add_test(NAME contact-evidence-snapshot COMMAND contact-evidence-snapshot-test)
  add_executable(grasp-attachment-core-test tests/grasp_attachment_core_test.cpp)
  target_compile_features(grasp-attachment-core-test PRIVATE cxx_std_17)
  target_link_libraries(grasp-attachment-core-test PRIVATE ignition-gazebo6::ignition-gazebo6)
  add_test(NAME grasp-attachment-core COMMAND grasp-attachment-core-test)
endif()


install(
  DIRECTORY
    config
    launch
    worlds
    models
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/target_cube_manager.py
    scripts/goal_region_pose_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES
    scripts/goal_region_pose_core.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
