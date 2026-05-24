add_library(core)
target_sources(core
                PUBLIC
                    ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/dom.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/to_ssa.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/leave_ssa.cpp
)
target_link_libraries(core PUBLIC Boost::boost util PRIVATE compiler_opts)
target_include_directories(core PUBLIC ${CMAKE_SOURCE_DIR}/src)

add_library(conv)
target_sources(conv
                PUBLIC
                    ${CMAKE_CURRENT_SOURCE_DIR}/parse.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/cfg.cpp
)
target_link_libraries(conv PUBLIC core nlohmann_json::nlohmann_json PRIVATE compiler_opts)
target_include_directories(conv PUBLIC ${CMAKE_SOURCE_DIR}/src)

add_library(brili-lib)
target_sources(brili-lib
                PUBLIC
                    ${CMAKE_CURRENT_SOURCE_DIR}/brili.cpp
)
target_link_libraries(brili-lib PUBLIC core PRIVATE compiler_opts)
target_include_directories(brili-lib PUBLIC ${CMAKE_SOURCE_DIR}/src)

add_subdirectory(util)
