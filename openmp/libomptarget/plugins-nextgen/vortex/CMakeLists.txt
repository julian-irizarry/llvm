##===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# Build a plugin for a VORTEX machine.
#
##===----------------------------------------------------------------------===##

set(LIBOMPTARGET_BUILD_VORTEX_PLUGIN TRUE CACHE BOOL
  "Whether to build VORTEX plugin")

if (NOT LIBOMPTARGET_BUILD_VORTEX_PLUGIN)
  libomptarget_say("Not building VORTEX offloading plugin: LIBOMPTARGET_BUILD_VORTEX_PLUGIN is false")
  return()
endif()

if (NOT (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86_64)|(ppc64le)|(aarch64)$" AND CMAKE_SYSTEM_NAME MATCHES "Linux"))
  libomptarget_say("Not building VORTEX plugin: only supported on Linux x86_64, ppc64le, or aarch64 hosts.")
  return()
endif()

libomptarget_say("Building VORTEX offloading plugin.")

add_llvm_library(omptarget.rtl.vortex SHARED
  src/rtl.cpp

  LINK_COMPONENTS
  Support
  Object

  LINK_LIBS PRIVATE
  PluginCommon
  ${OPENMP_PTHREAD_LIB}

  NO_INSTALL_RPATH
)

if ((OMPT_TARGET_DEFAULT) AND (LIBOMPTARGET_OMPT_SUPPORT))
  target_link_libraries(omptarget.rtl.vortex PRIVATE OMPT)
endif()

if (LIBOMP_HAVE_VERSION_SCRIPT_FLAG)
  target_link_libraries(omptarget.rtl.vortex PRIVATE
  "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/../exports,-z,defs")
endif()

if(LIBOMPTARGET_DEP_VORTEX_FOUND)
  libomptarget_say("Building VORTEX plugin linked against libvortex")
  target_link_libraries(omptarget.rtl.vortex PRIVATE VORTEX::vortex_driver)
else()
  libomptarget_say("Building VORTEX plugin using dynamic library loading")
  target_include_directories(omptarget.rtl.vortex PRIVATE dynamic_vortex)
  target_sources(omptarget.rtl.vortex PRIVATE dynamic_vortex/vortex.cpp)
endif()

# Set debug prefix and include directories
target_compile_definitions(omptarget.rtl.vortex PRIVATE TARGET_NAME="VORTEX")
target_compile_definitions(omptarget.rtl.vortex PRIVATE DEBUG_PREFIX="TARGET VORTEX RTL")
target_include_directories(omptarget.rtl.vortex PRIVATE ${LIBOMPTARGET_INCLUDE_DIR})

# Install plugin under the lib destination folder
install(TARGETS omptarget.rtl.vortex LIBRARY DESTINATION "${OPENMP_INSTALL_LIBDIR}")
set_target_properties(omptarget.rtl.vortex PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_RPATH "$ORIGIN:${CMAKE_CURRENT_BINARY_DIR}/.."
  CXX_VISIBILITY_PRESET protected)

