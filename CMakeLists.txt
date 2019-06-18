cmake_minimum_required(VERSION 3.14)
project(ArrayFireExample)

set(CMAKE_CXX_STANDARD 14)


include_directories(.)
include_directories(rapidxml)
FIND_PACKAGE(ArrayFire REQUIRED)
INCLUDE_DIRECTORIES(${ArrayFire_INCLUDE_DIRS})


add_executable(ArrayFireExample
        rapidxml/rapidxml.hpp
        rapidxml/rapidxml_iterators.hpp
        rapidxml/rapidxml_print.hpp
        rapidxml/rapidxml_utils.hpp
        AFCSVParser.cpp
        AFCSVParser.hpp
        CSVObject.cpp
        CSVObject.hpp
        main.cpp
        RapidxmlTest.cpp
        RapidxmlTest.hpp
        Utils.cpp
        Utils.hpp)

TARGET_LINK_LIBRARIES(ArrayFireExample ${ArrayFire_Unified_LIBRARIES} )