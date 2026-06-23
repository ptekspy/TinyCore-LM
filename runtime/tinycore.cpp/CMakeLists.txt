cmake_minimum_required(VERSION 3.20)
project(tinycore_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

add_library(tinycore-native-runtime OBJECT src/native_runtime.cpp)

add_executable(tinycore-inspect-manifest src/manifest_inspect.cpp $<TARGET_OBJECTS:tinycore-native-runtime>)
add_executable(tinycore-generate src/native_generate.cpp $<TARGET_OBJECTS:tinycore-native-runtime>)
add_executable(tinycore-native-tests tests/native_runtime_tests.cpp $<TARGET_OBJECTS:tinycore-native-runtime>)
target_compile_definitions(tinycore-native-tests PRIVATE TINYCORE_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_test(NAME tinycore-native-runtime COMMAND tinycore-native-tests)
