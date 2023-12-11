# # CORE library contains core types and some analyses
# add_library(core)
# target_sources(core 
#                 PUBLIC 
#                     ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
#                     ${CMAKE_CURRENT_SOURCE_DIR}/edit.cpp
#                     ${CMAKE_CURRENT_SOURCE_DIR}/backedvec.cpp
# )
# target_link_libraries(core PUBLIC Boost::boost PRIVATE compiler_opts)
# target_include_directories(core PUBLIC ${CMAKE_SOURCE_DIR}/src)

# add_library(conv)
# target_sources(conv 
#                 PUBLIC 
#                     ${CMAKE_CURRENT_SOURCE_DIR}/conv.cpp
#                     # ${CMAKE_CURRENT_SOURCE_DIR}/cfg.cpp
# )
# target_link_libraries(conv PUBLIC core nlohmann_json::nlohmann_json PRIVATE compiler_opts)
# target_include_directories(conv PUBLIC ${CMAKE_SOURCE_DIR}/src)

add_library(core)
target_sources(core
                PUBLIC
                    ${CMAKE_CURRENT_SOURCE_DIR}/types.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/small_vector.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/dom.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/to_ssa.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/leave_ssa.cpp
)
target_link_libraries(core PUBLIC Boost::boost PRIVATE compiler_opts)
target_include_directories(core PUBLIC ${CMAKE_SOURCE_DIR}/src)

add_library(conv)
target_sources(conv
                PUBLIC
                    ${CMAKE_CURRENT_SOURCE_DIR}/parse.cpp
                    ${CMAKE_CURRENT_SOURCE_DIR}/cfg.cpp
)
target_link_libraries(conv PUBLIC core nlohmann_json::nlohmann_json PRIVATE compiler_opts)
target_include_directories(conv PUBLIC ${CMAKE_SOURCE_DIR}/src)
