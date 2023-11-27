# CORE library contains core types and some analyses
add_library(core)
target_sources(core 
                PUBLIC 
                    ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
)
target_link_libraries(core PUBLIC Boost::boost)
target_include_directories(core PUBLIC ${CMAKE_SOURCE_DIR}/src)
