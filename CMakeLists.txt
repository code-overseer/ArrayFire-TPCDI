cmake_minimum_required(VERSION 3.10)
if( USING_CUDA AND USING_OPENCL )
    error( "ERROR: Cannot have CUDA and OpenCL beckend active at the same time" )
endif()

if (USING_CUDA)
    project(ArrayFire-TPCDI LANGUAGES CXX CUDA)
    set(CMAKE_CUDA_FLAGS "-Xptxas -dlcm=ca -arch=sm_61")
    add_definitions( -DUSING_CUDA )
    message(${CMAKE_CUDA_COMPILER})
    message("Using CUDA backend")
else()
    project(ArrayFire-TPCDI LANGUAGES CXX)
endif()

if (APPLE)
    add_definitions( -DIS_APPLE )
    add_definitions( -DCL_SILENCE_DEPRECATION )
    add_definitions( -DCL_LOG_ERRORS=stderr )
else()
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -gdwarf-3")
    if ( NOT DISABLE_ITT )
        include( "${CMAKE_SOURCE_DIR}/FindITT.cmake" )
    endif()
endif()

set(CMAKE_CXX_STANDARD 14)
FIND_PACKAGE(ArrayFire REQUIRED)
FIND_PACKAGE(Boost COMPONENTS filesystem regex REQUIRED)
INCLUDE_DIRECTORIES(include)
INCLUDE_DIRECTORIES(rapidxml)
INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})

if (USING_OPENCL)
    add_definitions( -DUSING_OPENCL )
    FIND_PACKAGE(OpenCL REQUIRED)
    INCLUDE_DIRECTORIES( ${OpenCL_INCLUDE_DIRS})
    message("Using OpenCL backend")
endif()

add_executable(ArrayFire-TPCDI
        rapidxml/rapidxml.hpp
        rapidxml/rapidxml_iterators.hpp
        rapidxml/rapidxml_print.hpp
        rapidxml/rapidxml_utils.hpp
        src/main.cpp
        src/AFParser.cpp
        src/AFDataFrame.cpp
        src/AFHashTable.cpp
        src/Column.cpp
        src/BatchFunctions.cpp
        src/FinwireParser.cpp
        src/Logger.cpp
        src/Tests.cpp
        src/TPCDI.cpp
        src/Utils.cpp
        src/Kernels/CPUSingleThreaded.cpp
        src/Kernels/KernelInterface.cpp
        include/AFParser.h
        include/Enums.h
        include/BatchFunctions.h
        include/Tests.h
        include/AFDataFrame.h
        include/TPCDI.h
        include/FinwireParser.h
        include/Utils.h
        include/Logger.h
        include/AFTypes.h
        include/Column.h
        include/ColumnNames.h
        include/AFHashTable.h
        include/Kernels.h
        include/KernelInterface.h)

if (ITT_FOUND)
   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lm")
   TARGET_LINK_LIBRARIES( ArrayFire-TPCDI libittnotify.a ${CMAKE_DL_LIBS} )
endif()

if (USING_CUDA)
    set_source_files_properties(src/AFDataFrame.cpp
            src/Tests.cpp
            src/main.cpp
            src/AFParser.cpp
            src/FinwireParser.cpp
            src/Column.cpp
            src/TPCDI.cpp
            src/Kernels/KernelInterface.cpp
            src/Kernels/CUDAKernels.cu
            PROPERTIES LANGUAGE CUDA)
    TARGET_SOURCES(ArrayFire-TPCDI PRIVATE src/Kernels/CUDAKernels.cu)
    set_property(TARGET ArrayFire-TPCDI PROPERTY CUDA_SEPARABLE_COMPILATION ON)
elseif(USING_OPENCL)
    add_definitions( -DOCL_KERNEL_DIR="${CMAKE_SOURCE_DIR}/src/Kernels/kernels.cl" )
    TARGET_SOURCES(ArrayFire-TPCDI PRIVATE src/Kernels/OpenCLKernels.cpp)
    TARGET_LINK_LIBRARIES(ArrayFire-TPCDI ${OpenCL_LIBRARIES} )
else()
    message("Using single-threaded CPU backend")
endif()

if (USING_AF)
    add_definitions( -DUSING_AF )
    message("Using ArrayFire operations")
else()
    message("Using hand-written kernels/functions")
endif()

TARGET_LINK_LIBRARIES(ArrayFire-TPCDI ${ArrayFire_Unified_LIBRARIES} )
TARGET_LINK_LIBRARIES(ArrayFire-TPCDI ${Boost_LIBRARIES} )
