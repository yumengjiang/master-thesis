cmake_minimum_required(VERSION 2.8)
project( main )
set(CMAKE_CXX_FLAGS "-std=c++11")
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
add_library(tinydir INTERFACE)
add_library(inviwo::tinydir ALIAS INTERFACE
TARGET_INCLUDE_DIRECTORIES(tinydir, INTERFACE
$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/>
)
ADD_CUSTOM_TARGET(tinydircpp SOURCE ${HEADER_FILES}}
set(CMAKE_BUILD_TYPE "RELEASE")
include_directories(${CERES_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${tinydir_INCLUDE_DIRS)
file(GLOB_RECURSE CURRENT_HEADERS  include/*.h include/*.hpp)
file(GLOB CURRENT_SOURCES  src/*.c src/*.cpp)
source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES})
add_executable(main main.cpp)
add_executable(reconstruction reconstruction.cpp)
# target_link_libraries(${CERES)
<<<<<<< HEAD
target_link_libraries( main ${CERES_LIBRARIES} ${OpenCV_LIBS} gflags )
target_link_libraries( reconstruction ${CERES_LIBRARIES} ${OpenCV_LIBS} gflags )
=======
target_link_libraries( main ${CERES_LIBRARIES} ${OpenCV_LIBS} ${tinydir_LIBS} gflags )
>>>>>>> 45635cf3747d9bba2562ca39089ff15af5270ded
