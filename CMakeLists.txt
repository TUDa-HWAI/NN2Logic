cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(nn2logic LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ version selection")  # or 11, 14, 17, 20
set(CMAKE_CXX_STANDARD_REQUIRED ON)  # optional, ensure standard is supported
set(CMAKE_CXX_EXTENSIONS OFF)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")


set(GUROBI_DIR "/Library/gurobi1101/macos_universal2" "~/gurobi1101/linux64")  # FIXME

# otherwise FMT will cry
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# CPM
set(CPM_DOWNLOAD_VERSION 0.38.2)  # Download CPM package manager
set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${CPM_DOWNLOAD_VERSION}.cmake")
if(NOT (EXISTS ${CPM_DOWNLOAD_LOCATION}))
  message(STATUS "Downloading CPM.cmake to ${CPM_DOWNLOAD_LOCATION}")
  file(DOWNLOAD
       https://github.com/cpm-cmake/CPM.cmake/releases/download/v${CPM_DOWNLOAD_VERSION}/CPM.cmake
       ${CPM_DOWNLOAD_LOCATION}
  )
endif()
include(${CPM_DOWNLOAD_LOCATION})

# add pybind11
CPMAddPackage("gh:pybind/pybind11@3.0.0")
pybind11_add_module(nn2logic)
install(TARGETS nn2logic DESTINATION ${CMAKE_CURRENT_SOURCE_DIR})

# indicators
CPMAddPackage("gh:p-ranav/indicators#9c855c9")
target_link_libraries(nn2logic PRIVATE indicators::indicators)

# fmt
CPMAddPackage("gh:fmtlib/fmt#12.1.0")
target_link_libraries(nn2logic PRIVATE fmt::fmt)
#add_compile_definitions(FMT_HEADER_ONLY)

# Eigen
find_package(Eigen3 3.4...5.0.1 REQUIRED NO_MODULE)
target_link_libraries(nn2logic PRIVATE Eigen3::Eigen)

# Json
CPMAddPackage("gh:nlohmann/json@3.11.3")
include_directories(${nlohmann_json_SOURCE_DIR}/include)
target_link_libraries(nn2logic PRIVATE nlohmann_json::nlohmann_json)

CPMAddPackage("gh:pybind/pybind11_json#0.2.14")
include_directories(${pybind11_json_SOURCE_DIR}/include)

FIND_PACKAGE(Boost 1.40 COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

# Gurobi
find_package(GUROBI REQUIRED)
include_directories(${GUROBI_INCLUDE_DIRS})
target_link_libraries(nn2logic PRIVATE ${GUROBI_CXX_LIBRARY} ${GUROBI_LIBRARY})

# cpu count
include(ProcessorCount)
ProcessorCount(N)
add_compile_definitions(NUM_CPU=${N})

# tbb
find_package(TBB COMPONENTS tbb)
if (NOT APPLE)
  target_link_libraries(nn2logic PRIVATE tbb)
ENDIF()

# template
file(COPY src/sim/template DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
target_compile_definitions(nn2logic PUBLIC BUILD_DIR="${CMAKE_CURRENT_BINARY_DIR}")

target_include_directories(nn2logic PRIVATE src)
add_subdirectory(src)