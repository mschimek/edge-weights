cmake_minimum_required(VERSION 3.14)

project(generate-edgeweights LANGUAGES CXX)

include(FetchContent)

FetchContent_Declare(
  kagen
  GIT_REPOSITORY https://github.com/KarlsruheGraphGeneration/KaGen.git
  GIT_TAG 13c98fe
  SYSTEM)

FetchContent_Declare(
  cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG v2.3.2
  SYSTEM)

FetchContent_Declare(
  ips4o
  GIT_REPOSITORY https://github.com/mschimek/ips4o.git
  GIT_TAG eb30352
  SYSTEM)

FetchContent_MakeAvailable(kagen cli11 ips4o)
find_package(MPI REQUIRED)
add_executable(generator src/generator.cpp)

target_link_libraries(generator PUBLIC MPI::MPI_CXX KaGen::KaGen CLI11::CLI11
                                        ips4o)
