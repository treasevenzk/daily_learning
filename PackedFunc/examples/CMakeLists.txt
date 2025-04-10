# Examples CMakeLists.txt
add_executable(basic_example basic_example.cpp)
target_link_libraries(basic_example PRIVATE packedfun)

# Install examples
install(TARGETS basic_example
        RUNTIME DESTINATION bin/examples)