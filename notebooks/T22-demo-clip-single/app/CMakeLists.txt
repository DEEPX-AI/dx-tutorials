cmake_minimum_required(VERSION 3.14)
project(clip_single DESCRIPTION "DEEPX CLIP single-stream Qt5 demo")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)

find_package(OpenCV REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets)
find_package(ZLIB REQUIRED)

if(CROSS_COMPILE OR MSVC)
    find_library(DXRT_LIB dxrt HINTS ${DXRT_INSTALLED_DIR}/lib REQUIRED)
    set(DXRT_INCLUDE_DIR ${DXRT_INSTALLED_DIR}/include)
else()
    find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})
    set(DXRT_LIB dxrt)
endif()

find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include ${ONNXRUNTIME_ROOT}
          /usr/local/include /usr/include
    REQUIRED
)
find_library(ONNXRUNTIME_LIB
    NAMES onnxruntime
    HINTS ${ONNXRUNTIME_ROOT}/lib /usr/local/lib /usr/lib
    REQUIRED
)

if(NOT DEFINED PROJECT_ROOT_OVERRIDE)
    get_filename_component(PROJECT_ROOT_OVERRIDE "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
endif()

add_executable(clip_single
    main.cpp
    clip_tokenizer.cpp
    clip_tokenizer.hpp
)

target_compile_definitions(clip_single PRIVATE
    PROJECT_ROOT_DIR="${PROJECT_ROOT_OVERRIDE}"
)

target_include_directories(clip_single PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${DXRT_INCLUDE_DIR}
)

target_compile_options(clip_single PRIVATE -Wall -Wextra -O3)

target_link_libraries(clip_single PRIVATE
    ${OpenCV_LIBS}
    Qt5::Widgets
    ZLIB::ZLIB
    ${DXRT_LIB}
    ${ONNXRUNTIME_LIB}
    pthread
    dl
)
