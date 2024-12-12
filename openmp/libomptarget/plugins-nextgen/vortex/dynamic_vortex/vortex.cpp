//===--- vortex/dynamic_vortex/vortex.cpp ----------------------- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement subset of Vortex API by calling into Vortex library via dlopen
// Does the dlopen/dlsym calls as part of the initialization process.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/DynamicLibrary.h"

#include "Shared/Debug.h"

#include "DLWrap.h"
#include "vortex.h"

#include <memory>
#include <string>
#include <unordered_map>

DLWRAP_INITIALIZE()

// Device management
DLWRAP(vx_dev_open, 1)
DLWRAP(vx_dev_close, 1)
DLWRAP(vx_dev_caps, 3)

// Memory management
DLWRAP(vx_mem_alloc, 4)
DLWRAP(vx_mem_reserve, 5)
DLWRAP(vx_mem_free, 1)
DLWRAP(vx_mem_access, 4)
DLWRAP(vx_mem_address, 2)
DLWRAP(vx_mem_info, 3)

// Data transfer
DLWRAP(vx_copy_to_dev, 4)
DLWRAP(vx_copy_from_dev, 4)

// Execution
DLWRAP(vx_start, 3)
DLWRAP(vx_ready_wait, 2)

// Performance and utilities
DLWRAP(vx_dcr_read, 3)
DLWRAP(vx_dcr_write, 3)
DLWRAP(vx_mpm_query, 4)
DLWRAP(vx_upload_kernel_bytes, 4)
DLWRAP(vx_upload_kernel_file, 3)
DLWRAP(vx_check_occupancy, 3)
DLWRAP(vx_dump_perf, 2)

DLWRAP_FINALIZE()

#ifndef DYNAMIC_VORTEX_PATH
#define DYNAMIC_VORTEX_PATH "libvortex.so"
#endif

#ifndef TARGET_NAME
#define TARGET_NAME VORTEX
#endif
#ifndef DEBUG_PREFIX
#define DEBUG_PREFIX "Target " GETNAME(TARGET_NAME) " RTL"
#endif

static bool checkForVortex() {
  const char *VortexLib = DYNAMIC_VORTEX_PATH; // Path to the Vortex shared library
  std::string ErrMsg;
  auto DynlibHandle = std::make_unique<llvm::sys::DynamicLibrary>(
      llvm::sys::DynamicLibrary::getPermanentLibrary(VortexLib, &ErrMsg));
  if (!DynlibHandle->isValid()) {
    DP("Unable to load library '%s': %s!\n", VortexLib, ErrMsg.c_str());
    return false;
  }

  for (size_t I = 0; I < dlwrap::size(); I++) {
    const char *Sym = dlwrap::symbol(I);
    void *P = DynlibHandle->getAddressOfSymbol(Sym);
    if (P == nullptr) {
      DP("Unable to find '%s' in '%s'!\n", Sym, VortexLib);
      return false;
    }
    DP("Implementing %s with dlsym(%s) -> %p\n", Sym, Sym, P);
    *dlwrap::pointer(I) = P;
  }

  return true;
}
