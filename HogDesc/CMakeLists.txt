cmake_minimum_required(VERSION 3.23)
project(HogDesc)
set(CMAKE_CXX_STANDARD 14)
find_package(OpenCV REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS})
aux_source_directory(./ FILES)
add_executable(HogDesc ${FILES})
target_link_libraries(HogDesc ${OpenCV_LIBS})

