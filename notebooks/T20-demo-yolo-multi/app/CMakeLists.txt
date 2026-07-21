cmake_minimum_required(VERSION 3.14)
project(yolo_multi_demo DESCRIPTION "DEEPX YOLO Multi-Channel Object Detection Demo")

add_compile_definitions(PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

if(NOT "${CMAKE_HOST_SYSTEM_PROCESSOR}" STREQUAL "${CMAKE_SYSTEM_PROCESSOR}")
    set(CROSS_COMPILE TRUE)
else()
    set(CROSS_COMPILE FALSE)
endif()

set(STD_FS_NO_LIB_NEEDED FALSE)
if(CMAKE_CXX_STANDARD GREATER_EQUAL 17)
    set(STD_FS_NO_LIB_NEEDED TRUE)
endif()

include(dxdemo.cfg)
include(dxdemo.function)

execute_process(
    COMMAND gst-inspect-1.0 vaapidecodebin
    RESULT_VARIABLE GST_VAAPI_FOUND
    OUTPUT_QUIET ERROR_QUIET
)
if(GST_VAAPI_FOUND EQUAL 0)
    add_definitions(-DUSE_VAAPI)
endif()

add_compile_options(-W -Wall -pthread -fstrict-volatile-bitfields -fPIC -march=native -fopenmp)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fopenmp")
if(ENABLE_DEBUG_INFO)
    add_compile_options(-g -rdynamic)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

add_dxrt_lib()
add_opencv()

# Check for OpenCV freetype contrib module
if(TARGET opencv_freetype OR OpenCV_freetype_FOUND OR "opencv_freetype" IN_LIST OpenCV_LIBS)
  set(HAVE_OPENCV_FREETYPE TRUE)
else()
  find_library(_opencv_freetype_lib NAMES opencv_freetype PATHS ${OpenCV_INSTALL_PATH}/lib NO_DEFAULT_PATH)
  if(_opencv_freetype_lib)
    set(HAVE_OPENCV_FREETYPE TRUE)
  endif()
endif()
if(NOT HAVE_OPENCV_FREETYPE)
  find_package(Freetype)
  if(Freetype_FOUND)
    set(HAVE_FREETYPE_DIRECT TRUE)
  endif()
endif()

set(DEMO_UTILS_DIR ${CMAKE_SOURCE_DIR}/src/demo_utils)
set(DEMO_UTILS_SRC
    ${DEMO_UTILS_DIR}/yolo_cfg.cpp
    ${DEMO_UTILS_DIR}/yolo.cpp
    ${DEMO_UTILS_DIR}/nms.cpp
    ${DEMO_UTILS_DIR}/bbox.cpp
    ${DEMO_UTILS_DIR}/image.cpp
    ${DEMO_UTILS_DIR}/display.cpp
)

add_executable(yolo_multi_demo
    ${DEMO_UTILS_SRC}
    ${CMAKE_SOURCE_DIR}/src/od.cpp
    ${CMAKE_SOURCE_DIR}/src/yolo_multi_channels.cpp
)

if(STD_FS_NO_LIB_NEEDED)
    set(STD_FS_LIB "")
else()
    set(STD_FS_LIB stdc++fs)
endif()

target_include_directories(yolo_multi_demo PRIVATE
    ${DEMO_UTILS_DIR}/include
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/extern/
    ${CMAKE_SOURCE_DIR}/lib/
)
target_link_libraries(yolo_multi_demo ${link_libs} ${STD_FS_LIB})

if(HAVE_OPENCV_FREETYPE)
  target_compile_definitions(yolo_multi_demo PRIVATE HAVE_OPENCV_FREETYPE)
endif()
if(HAVE_FREETYPE_DIRECT)
  target_compile_definitions(yolo_multi_demo PRIVATE HAVE_FREETYPE_DIRECT)
  target_link_libraries(yolo_multi_demo Freetype::Freetype)
endif()

install(TARGETS yolo_multi_demo DESTINATION bin)
