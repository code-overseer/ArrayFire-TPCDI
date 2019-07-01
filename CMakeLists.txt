cmake_minimum_required(VERSION 3.10)
project(ArrayFire-TPCDI)

set(CMAKE_CXX_STANDARD 14)


include_directories(.)
include_directories(rapidxml)
FIND_PACKAGE(ArrayFire REQUIRED)
INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})


add_executable(ArrayFire-TPCDI
        rapidxml/rapidxml.hpp
        rapidxml/rapidxml_iterators.hpp
        rapidxml/rapidxml_print.hpp
        rapidxml/rapidxml_utils.hpp
        AFParser.cpp
        AFParser.hpp
        main.cpp
        Enums.h BatchFunctions.cpp BatchFunctions.h Tests.cpp Tests.h AFDataFrame.cpp AFDataFrame.h TPC_DI.cpp TPC_DI.h XMLFlattener.cpp XMLFlattener.h)

TARGET_LINK_LIBRARIES(ArrayFire-TPCDI ${ArrayFire_Unified_LIBRARIES} )
