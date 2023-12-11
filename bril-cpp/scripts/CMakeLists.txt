add_executable(parse-end-to-end parse_end_to_end.cpp)
target_link_libraries(parse-end-to-end PRIVATE core conv compiler_opts)

add_executable(print-dom print_dom.cpp)
target_link_libraries(print-dom PRIVATE core conv compiler_opts)

add_executable(ssa-conv ssa_conv.cpp)
target_link_libraries(ssa-conv PRIVATE core conv compiler_opts)
