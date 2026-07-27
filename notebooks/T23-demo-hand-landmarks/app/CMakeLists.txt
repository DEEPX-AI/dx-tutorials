cmake_minimum_required(VERSION 3.14)
project(hand_landmarks DESCRIPTION "Hand landmarks Qt5 demo")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)

if(CROSS_COMPILE OR MSVC)
    find_library(DXRT_LIB dxrt HINTS ${DXRT_INSTALLED_DIR}/lib REQUIRED)
    set(DXRT_INCLUDE_DIR ${DXRT_INSTALLED_DIR}/include)
else()
    find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})
    set(DXRT_LIB dxrt)
endif()

if(NOT DEFINED PROJECT_ROOT_OVERRIDE)
    get_filename_component(PROJECT_ROOT_OVERRIDE "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
endif()

add_executable(hand_landmarks
    hand_landmarks.cpp
)

target_compile_definitions(hand_landmarks PRIVATE
    PROJECT_ROOT_DIR="${PROJECT_ROOT_OVERRIDE}"
)

target_include_directories(hand_landmarks PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${DXRT_INCLUDE_DIR}
)

target_compile_options(hand_landmarks PRIVATE -Wall -Wextra -O3)

target_link_libraries(hand_landmarks PRIVATE
    ${OpenCV_LIBS}
    Qt5::Widgets
    ${DXRT_LIB}
    pthread
)
