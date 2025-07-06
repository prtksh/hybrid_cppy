cmake_minimum_required(VERSION 3.12)
project(HybridInference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Fetch pybind11 (portable + recommended)
include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1  # Use latest stable or desired version
)
FetchContent_MakeAvailable(pybind11)

# Find Python3 and NumPy
find_package(Python3 REQUIRED COMPONENTS Interpreter Development NumPy)

# Source files
set(SOURCES
    src/main.cpp
    src/tensor_engine.cpp
)

# Include headers from src/
include_directories(src)

# Create executable
add_executable(hybrid_inference ${SOURCES})



# Link with pybind11 and Python
target_link_libraries(hybrid_inference PRIVATE
    pybind11::embed
    Python3::Python
    Python3::NumPy
)

# Optimization flags
target_compile_options(hybrid_inference PRIVATE
    $<$<CONFIG:Release>:-O3 -march=native>
    $<$<CONFIG:Debug>:-g -O0>
    -fstack-protector-strong
    -D_FORTIFY_SOURCE=2
)

# Ensure Python libraries are correctly linked at runtime
set_target_properties(hybrid_inference PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)

