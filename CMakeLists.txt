cmake_minimum_required(VERSION 2.8)
<<<<<<< HEAD
project( gatherpoints )
=======
project( sift_cnn )
>>>>>>> 64494008deeb3605855504907b9870f0ea2e2c38
set(CMAKE_CXX_FLAGS "-std=c++11")
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
set(CMAKE_BUILD_TYPE "RELEASE")
include_directories(${CERES_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
file(GLOB_RECURSE CURRENT_HEADERS  include/*.h include/*.hpp)
file(GLOB CURRENT_SOURCES  src/*.c src/*.cpp)
source_group("Include" FILES ${CURRENT_HEADERS}) 
source_group("Source" FILES ${CURRENT_SOURCES})
<<<<<<< HEAD
add_executable(gatherpoints gatherpoints.cpp)
# target_link_libraries(${CERES)
target_link_libraries( gatherpoints ${CERES_LIBRARIES} ${OpenCV_LIBS} gflags )
=======
add_executable(sift_cnn sift_cnn.cpp)
# target_link_libraries(${CERES)
target_link_libraries( sift_cnn ${CERES_LIBRARIES} ${OpenCV_LIBS} gflags )
>>>>>>> 64494008deeb3605855504907b9870f0ea2e2c38

