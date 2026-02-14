add_library(util)
target_sources(util
                PUBLIC 
                    ${CMAKE_CURRENT_SOURCE_DIR}/small_vector.cpp
)
target_link_libraries(util PRIVATE compiler_opts)
target_include_directories(util PUBLIC ${CMAKE_SOURCE_DIR}/src)
