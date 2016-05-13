cmake_minimum_required(VERSION 3.5)
project(calib_test)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenCV 2.4.9 EXACT REQUIRED)

set(SOURCE_FILES src/main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(main ${SOURCE_FILES})
target_link_libraries(main ${OpenCV_LIBRARIES})