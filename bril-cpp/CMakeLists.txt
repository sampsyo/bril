cmake_minimum_required(VERSION 3.16)

project(brilcpp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

include(FetchContent)

########
# JSON #
########

FetchContent_Declare(
    json 
    URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
    DOWNLOAD_EXTRACT_TIMESTAMP YES
)
FetchContent_MakeAvailable(json)
set(JSON_ImplicitConversions off)


#########
# BOOST #
#########

find_package(Boost REQUIRED)

####################
# COMPILER OPTIONS #
####################

add_library(compiler_opts INTERFACE)
if (MSVC)
    target_compile_options(compiler_opts INTERFACE /W4 /w14640 /permissive-)
else ()
    target_compile_options(compiler_opts INTERFACE -Wall -Werror -Wextra -Wshadow 
                                            -Wnon-virtual-dtor -Wconversion 
                                            -Wsign-conversion -Wnull-dereference 
                                            -pedantic)
endif ()
target_compile_options(compiler_opts INTERFACE $<$<CONFIG:Debug>:-DETAC_DEBUG>)

##################
# SUBDIRECTORIES #
##################

add_subdirectory(src)
add_subdirectory(scripts)
