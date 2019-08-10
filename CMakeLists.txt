cmake_minimum_required(VERSION 3.10)
if (USING_CUDA)
    project(ArrayFire-TPCDI LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_FLAGS -arch=sm_30)
    add_definitions( -DUSING_CUDA )
else()
    project(ArrayFire-TPCDI LANGUAGES CXX)
endif()

if (USING_AF)
    add_definitions( -DUSING_AF)
endif()

if (APPLE)
    add_definitions( -DIS_APPLE )
    add_definitions( -DCL_SILENCE_DEPRECATION )
    add_definitions( -DCL_LOG_ERRORS=stderr )
else()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")
endif()
set(CMAKE_CXX_STANDARD 14)
include_directories(.)
include_directories(rapidxml)
FIND_PACKAGE(ArrayFire REQUIRED)
INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})
FIND_PACKAGE(Boost COMPONENTS filesystem regex REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
if (USING_OPENCL)
    add_definitions( -DUSING_OPENCL )
    FIND_PACKAGE(OpenCL REQUIRED)
    INCLUDE_DIRECTORIES( ${OpenCL_INCLUDE_DIRS})
endif()

add_executable(ArrayFire-TPCDI
        rapidxml/rapidxml.hpp
        rapidxml/rapidxml_iterators.hpp
        rapidxml/rapidxml_print.hpp
        rapidxml/rapidxml_utils.hpp
        src/main.cpp
        src/AFParser.cpp
        src/BatchFunctions.cpp
        src/Tests.cpp
        src/AFDataFrame.cpp
        src/TPCDI.cpp
        src/Logger.cpp
        src/FinwireParser.cpp
        src/TPCDI_Utils.cpp
        src/Column.cpp
        include/AFParser.hpp
        include/Enums.h
        include/BatchFunctions.h
        include/Tests.h
        include/AFDataFrame.h
        include/TPCDI.h
        include/FinwireParser.h
        include/TPCDI_Utils.h
        include/Logger.h
        include/CPU/vector_functions.h
        include/AFTypes.h
        include/Column.h)

if (USING_CUDA)
    set_source_files_properties(src/AFDataFrame.cpp
            src/Tests.cpp
            src/main.cpp
            src/AFParser.cpp
            src/Column.cpp
            src/FinwireParser.cpp
            src/TPCDI.cpp
            include/CUDA/cuda_kernels.h
            PROPERTIES LANGUAGE CUDA)
    TARGET_SOURCES(ArrayFire-TPCDI PRIVATE include/CUDA/cuda_kernels.h)
    set_property(TARGET ArrayFire-TPCDI PROPERTY CUDA_SEPARABLE_COMPILATION ON)
elseif(USING_OPENCL)
    add_definitions( -DOCL_KERNEL_DIR="${CMAKE_SOURCE_DIR}/include/OpenCL/" )
    TARGET_SOURCES(ArrayFire-TPCDI PRIVATE include/OpenCL/opencl_kernels.h
            include/OpenCL/opencl_helper.h
            include/OpenCL/opencl_parsers.h)
    TARGET_LINK_LIBRARIES(ArrayFire-TPCDI ${OpenCL_LIBRARIES} )
endif()

TARGET_LINK_LIBRARIES(ArrayFire-TPCDI ${ArrayFire_Unified_LIBRARIES} )
TARGET_LINK_LIBRARIES(ArrayFire-TPCDI ${Boost_LIBRARIES} )
