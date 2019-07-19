cmake_minimum_required(VERSION 2.8.12)
project(fdeta)
# Set source directory
set(SOURCE_DIR "fdeta")
# Tell CMake that headers are also in SOURCE_DIR
add_subdirectory(lib/pybind11)
pybind11_add_module(auxfns "${SOURCE_DIR}/fdetmd/auxfns.cpp")
pybind11_add_module(cgrid_tools "${SOURCE_DIR}/fdetmd/cgrid_tools.cpp")
