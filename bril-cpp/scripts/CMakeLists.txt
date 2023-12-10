add_executable(parse_end_to_end parse_end_to_end.cpp)
target_link_libraries(parse_end_to_end PRIVATE core conv compiler_opts)

add_executable(print_dom print_dom.cpp)
target_link_libraries(print_dom PRIVATE core conv compiler_opts)
