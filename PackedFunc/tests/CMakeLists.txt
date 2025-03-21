# Tests CMakeLists.txt
add_executable(test_registry cpp/test_registry.cpp)
target_link_libraries(test_registry PRIVATE packedfun)

# Add test
add_test(NAME test_registry COMMAND test_registry)