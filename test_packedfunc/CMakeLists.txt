cmake_minimum_required(VERSION 3.10)
project(MyLib)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)

# 设置输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# 创建共享库
add_library(mylib SHARED
    mylib.cpp
)

# 设置符号可见性
target_compile_options(mylib PRIVATE -fvisibility=hidden)

# 设置RPATH
set_target_properties(mylib PROPERTIES
    SKIP_BUILD_RPATH FALSE
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
)