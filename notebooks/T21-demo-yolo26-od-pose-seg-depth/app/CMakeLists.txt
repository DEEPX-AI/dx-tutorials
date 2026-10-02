cmake_minimum_required(VERSION 3.14)
project(yolo26s_4_qt DESCRIPTION "YOLO26-S 4-task Qt demo")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

if(CROSS_COMPILE OR MSVC)
    find_library(DXRT_LIB dxrt HINTS ${DXRT_INSTALLED_DIR}/lib REQUIRED)
    include_directories(${DXRT_INSTALLED_DIR}/include)
    link_directories(${DXRT_INSTALLED_DIR}/lib)
else()
    find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})
    set(DXRT_LIB dxrt)
endif()

if(NOT DEFINED PROJECT_ROOT_OVERRIDE)
    get_filename_component(PROJECT_ROOT_OVERRIDE "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
endif()

add_executable(yolo26s_4
    yolo26s_4.cpp
)

target_compile_definitions(yolo26s_4 PRIVATE PROJECT_ROOT_DIR="${PROJECT_ROOT_OVERRIDE}")

target_include_directories(yolo26s_4 PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_compile_options(yolo26s_4 PRIVATE -Wall -Wextra -O3)

target_link_libraries(yolo26s_4
    Qt5::Widgets
    ${OpenCV_LIBS}
    ${DXRT_LIB}
)
