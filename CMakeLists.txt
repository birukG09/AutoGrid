cmake_minimum_required(VERSION 3.10)
project(AutoGridEmbedded)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(src)

# Add executable
add_executable(embedded_controller src/embedded_controller.cpp)

# Add shared library for Python FFI
add_library(embedded_controller_lib SHARED src/embedded_controller.cpp)

# Compiler flags for real-time performance
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -Wall -Wextra")

# Add threading support
find_package(Threads REQUIRED)
target_link_libraries(embedded_controller Threads::Threads)
target_link_libraries(embedded_controller_lib Threads::Threads)

# Set output names
set_target_properties(embedded_controller_lib PROPERTIES OUTPUT_NAME embedded_controller)