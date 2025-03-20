cmake_minimum_required(VERSION 3.10)
project(mylib VERSION 0.1.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 查找Python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

# 设置输出目录
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 包含目录
include_directories(include)

# 源文件
set(SOURCES
    src/api.cc
    src/runtime/packed_func.cc
    src/runtime/registry.cc
    src/python_api.cc
)

# 创建共享库
add_library(mylib SHARED ${SOURCES})

# 链接Python库
target_link_libraries(mylib ${PYTHON_LIBRARIES})

# 为Python绑定创建特殊的命名
if(MSVC)
    set_target_properties(mylib PROPERTIES OUTPUT_NAME "libmylib" PREFIX "" SUFFIX ".pyd")
else()
    set_target_properties(mylib PROPERTIES OUTPUT_NAME "libmylib" PREFIX "" SUFFIX ".so")
endif()

# 安装目标
install(TARGETS mylib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)

# 安装头文件
install(DIRECTORY include/ DESTINATION include)

# 输出库信息
get_target_property(MYLIB_LIBRARY_PATH mylib LIBRARY_OUTPUT_DIRECTORY)
message(STATUS "MyLib library will be built at: ${MYLIB_LIBRARY_PATH}")
message(STATUS "Python libraries: ${PYTHON_LIBRARIES}")