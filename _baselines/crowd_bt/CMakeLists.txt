cmake_minimum_required(VERSION 3.20)
project(repair_crowd_bt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
set(ABSL_PROPAGATE_CXX_STD ON CACHE BOOL "" FORCE)

include(FetchContent)
FetchContent_Declare(
  absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG        54fac219c4ef0bc379dfffb0b8098725d77ac81b
)
FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        b2814d53a707f699e5c56f565847e7020654efc2
)
FetchContent_Declare(
  highway
  GIT_REPOSITORY https://github.com/google/highway.git
  GIT_TAG        ba0900a4957b929390ab73827235557959234fea
)
FetchContent_MakeAvailable(absl eigen highway)

add_library(elo elo.cc elo_fast_math.cc)
target_include_directories(elo PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(
  elo PUBLIC
  Eigen3::Eigen absl::flat_hash_map absl::log absl::check
  absl::time absl::random_random hwy
)
add_executable(elo_main elo_main.cc)
target_link_libraries(elo_main PRIVATE elo absl::flags absl::flags_parse absl::log_initialize)
