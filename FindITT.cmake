# Copyright (c) 2017-2018 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# VTune is a source of ITT library
if( NOT CMAKE_VTUNE_HOME )
    set( CMAKE_VTUNE_HOME /opt/intel/vtune_amplifier )
endif()

find_path( ITT_INCLUDE_DIRS ittnotify.h
        PATHS ${CMAKE_ITT_HOME} ${CMAKE_VTUNE_HOME}
        PATH_SUFFIXES include )

find_path( ITT_LIBRARY_DIRS libittnotify.a
        PATHS ${CMAKE_ITT_HOME} ${CMAKE_VTUNE_HOME}
        PATH_SUFFIXES lib64 )

if(NOT ITT_INCLUDE_DIRS MATCHES NOTFOUND AND
        NOT ITT_LIBRARY_DIRS MATCHES NOTFOUND)
    message( STATUS "itt header is in ${ITT_INCLUDE_DIRS}" )
    message( STATUS "itt lib is in ${ITT_LIBRARY_DIRS}" )

    include_directories( ${ITT_INCLUDE_DIRS} )
    link_directories( ${ITT_LIBRARY_DIRS} )

    set( ITT_FOUND TRUE )
    add_definitions( -DENABLE_ITT )
endif()

if (NOT ITT_FOUND)
    message( "Failed to find ITT library" )
endif()

