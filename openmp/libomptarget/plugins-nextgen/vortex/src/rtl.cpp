//===--------- VortexPlugin.cpp - Vortex Plugin Implementation ------------===//
//
// This is an example implementation of a GenericPluginTy and GenericDeviceTy
// interface for a hypothetical Vortex GPGPU target, integrating with the
// Vortex runtime.
//
//===----------------------------------------------------------------------===//

#include <vortex.h>

#include "Shared/Debug.h"
#include "Shared/Environment.h"

#include "GlobalHandler.h"
#include "OpenMP/OMPT/Callback.h"
#include "PluginInterface.h"
#include "Utils/ELF.h"

#include "llvm/BinaryFormat/ELF.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"
#include "llvm/Frontend/OpenMP/OMPGridValues.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileOutputBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

// Include other necessary headers for ELF checking, error handling, etc.
namespace llvm {
namespace omp {
namespace target {
namespace plugin {

/// Forward declarations for all specialized data structures.
struct VortexKernelTy;
struct VortexDeviceTy;
struct VortexPluginTy;

////////////////////////////////////////////////////////////////////////////////
// VortexKernelTy - A Vortex kernel representation
////////////////////////////////////////////////////////////////////////////////
class VortexGlobalHandlerTy final : public GenericGlobalHandlerTy {
public:
  /// Get the metadata of a global from the device. The name and size of the
  /// global is read from DeviceGlobal and the address of the global is written
  /// to DeviceGlobal.
  Error getGlobalMetadataFromDevice(GenericDeviceTy &Device,
                                    DeviceImageTy &Image,
                                    GlobalTy &DeviceGlobal) override {
    return Plugin::success();
  }
};

struct VortexKernelTy : public GenericKernelTy {
  // For Vortex, a kernel might be represented by a device code handle,
  // or just by a symbol name plus some metadata.

  VortexKernelTy(const char *Name) : GenericKernelTy(Name) {}

  Error initImpl(GenericDeviceTy &GenericDevice,
                 DeviceImageTy &Image) override {
    // Typically, you'd locate the symbol for the kernel entry point here,
    // but Vortex uses vx_upload_kernel_file and vx_start.
    // This plugin model expects you to have some symbol. For simplicity,
    // assume the kernel is embedded in the Image and we do nothing special.
    return Plugin::success();
  }

  Error launchImpl(GenericDeviceTy &GenericDevice, uint32_t NumThreads,
                   uint64_t NumBlocks, KernelArgsTy &KernelArgs, void *Args,
                   AsyncInfoWrapperTy &AsyncInfoWrapper) const override {
    // In the example runtime code, launching a kernel involves:
    // 1. Uploading the kernel binary (already done on loadBinary)
    // 2. Uploading the arguments
    // 3. Starting and waiting for completion (which we do asynchronously here)

    // NOTE: The plugin interface separates launching from waiting. We may need
    // asynchronous variants if Vortex supports them. Otherwise, we might just
    // block (not ideal).

    // We'll assume that the device image and arguments are already set.
    // The Args pointer typically includes all kernel arguments.
    // For simplicity, suppose we have a structure representing them.

    // TODO: Implement the actual kernel launch.
    // On Vortex, from the provided example, you'd do something like:
    //   vx_start(deviceHandle, krnl_buffer, args_buffer);
    // However, we need a per-kernel resource handle stored somewhere.

    return Plugin::success();
  }
};

////////////////////////////////////////////////////////////////////////////////
// VortexDeviceTy - A Vortex device representation
////////////////////////////////////////////////////////////////////////////////

struct VortexDeviceTy : public GenericDeviceTy {
  vx_device_h DeviceHandle = nullptr;

  VortexDeviceTy(int32_t DeviceId, int32_t NumDevices)
      : GenericDeviceTy(DeviceId, NumDevices, NVPTXGridValues) {}

  Error setContext() override { return Plugin::success(); };

  // Device initialization
  Error initImpl(GenericPluginTy &Plugin) override {
    int ret = vx_dev_open(&DeviceHandle);
    if (ret != 0)
      return Plugin::error("Error in vx_dev_open: %d", ret);
    return Plugin::success();
  }

  // Device de-initialization
  Error deinitImpl() override {
    if (DeviceHandle) {
      vx_dev_close(DeviceHandle);
      DeviceHandle = nullptr;
    }
    return Plugin::success();
  }

  Expected<DeviceImageTy *> loadBinaryImpl(const __tgt_device_image *TgtImage,
                                           int32_t ImageId) override {
    // For Vortex, we can upload the binary using vx_upload_kernel_file.
    // The TgtImage->ImageStart/End points to the binary content.
    // We'll store the resulting handle in a DeviceImageTy.

    // The runtime expects a file, but we have a memory buffer.
    // If vx_upload_kernel_file does not support from-memory loading,
    // we might need to write the image to a temp file first.
    // Assume a hypothetical vx_upload_kernel_memory for demonstration,
    // or implement a temporary file solution.

    // Hypothetical code:
    // vx_buffer_h krnl_buffer;
    // int ret = vx_upload_kernel_memory(DeviceHandle,
    // (void*)TgtImage->ImageStart,
    //                                   (size_t)getPtrDiff(TgtImage->ImageEnd,
    //                                   TgtImage->ImageStart), &krnl_buffer);
    // if (ret != 0) return Plugin::error("Failed to upload kernel image");

    // Create the DeviceImageTy
    auto *Image =
        Plugin::get().allocate<DeviceImageTy>();
    return Image;
  }

  // Synchronize pending operations
  Error synchronizeImpl(__tgt_async_info &AsyncInfo) override {
    // If we rely on vx_ready_wait to ensure completion:
    // vx_ready_wait(DeviceHandle, VX_MAX_TIMEOUT);
    // But we should also consider AsyncInfo. If no asynchronous calls exist,
    // we can just block here.
    return Plugin::success();
  }

  Error queryAsyncImpl(__tgt_async_info &AsyncInfo) override {
    // Non-blocking query is not supported by the simple runtime example.
    // If vx_ready_wait can return immediately if complete, we could do that.
    // Otherwise just return not implemented or success.
    return Plugin::success();
  }

  Error getDeviceMemorySize(uint64_t &DSize) override {
    // If the device memory size is known or fixed, return it here.
    DSize = 0; // placeholder
    return Plugin::success();
  }

  Error memoryVAMap(void **Addr, void *VAddr, size_t *RSize) override {
    // If Vortex doesn't support advanced VA management, return error.
    return Plugin::error("memoryVAMap not supported");
  }

  Error memoryVAUnMap(void *VAddr, size_t Size) override {
    // If not supported
    return Plugin::error("memoryVAUnMap not supported");
  }

  void *allocate(size_t Size, void *HostPtr, TargetAllocTy Kind) override {
    vx_buffer_h buffer;
    int ret = vx_mem_alloc(
        DeviceHandle, (uint64_t)Size,
        (Kind == TARGET_ALLOC_DEFAULT || Kind == TARGET_ALLOC_DEVICE)
            ? VX_MEM_READ_WRITE
            : VX_MEM_READ,
        &buffer);
    if (ret != 0) {
      REPORT("Allocate failed with code %d\n", ret);
      return nullptr;
    }
    // Return the buffer handle as the device pointer
    return (void *)buffer;
  }

  int free(void *TgtPtr, TargetAllocTy Kind) override {
    vx_buffer_h buffer = (vx_buffer_h)TgtPtr;
    int ret = vx_mem_free(buffer);
    if (ret != 0) {
      REPORT("Free failed with code %d\n", ret);
      return OFFLOAD_FAIL;
    }
    return OFFLOAD_SUCCESS;
  }

  Expected<void *> dataLockImpl(void *HstPtr, int64_t Size) override {
    // If Vortex does not support host memory pinning, just return the original
    // pointer. Otherwise, implement pinning logic.
    return HstPtr;
  }

  Error dataUnlockImpl(void *HstPtr) override {
    // If pinning was done, unpin. Otherwise, just return success.
    return Plugin::success();
  }

  Error dataSubmitImpl(void *TgtPtr, const void *HstPtr, int64_t Size,
                       AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    vx_buffer_h buffer = (vx_buffer_h)TgtPtr;
    int ret = vx_copy_to_dev(buffer, HstPtr, 0, (size_t)Size);
    if (ret != 0)
      return Plugin::error("dataSubmit failed: %d", ret);
    return Plugin::success();
  }

  Error dataRetrieveImpl(void *HstPtr, const void *TgtPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    vx_buffer_h buffer = (vx_buffer_h)TgtPtr;
    int ret = vx_copy_from_dev(HstPtr, buffer, 0, (size_t)Size);
    if (ret != 0)
      return Plugin::error("dataRetrieve failed: %d", ret);
    return Plugin::success();
  }

  Error dataExchangeImpl(const void *SrcPtr, GenericDeviceTy &DstDev,
                         void *DstPtr, int64_t Size,
                         AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // If device-to-device copy is not supported, we can fallback to
    // host-staging or return an error.
    return Plugin::error("dataExchange not supported");
  }

  Error initAsyncInfoImpl(AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    // Vortex might not support separate asynchronous queues directly.
    // If none, we can just leave AsyncInfo->Queue = nullptr.
    return Plugin::success();
  }

  Error initDeviceInfoImpl(__tgt_device_info *DeviceInfo) override {
    // Initialize device info if needed.
    return Plugin::success();
  }

  Error createEventImpl(void **EventPtrStorage) override {
    // If no native events, return error or implement a dummy event.
    return Plugin::error("Events not supported");
  }

  Error destroyEventImpl(void *EventPtr) override { return Plugin::success(); }

  Error recordEventImpl(void *EventPtr,
                        AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::error("Events not supported");
  }

  Error waitEventImpl(void *EventPtr,
                      AsyncInfoWrapperTy &AsyncInfoWrapper) override {
    return Plugin::error("Events not supported");
  }

  Error syncEventImpl(void *EventPtr) override {
    return Plugin::error("Events not supported");
  }

  Error obtainInfoImpl(InfoQueueTy &Info) override {
    // Provide any device specific info.
    Info.add("Device", "Vortex");
    return Plugin::success();
  }

  Error getDeviceStackSize(uint64_t &V) override {
    // If stack size is configurable or known, return it. Otherwise, zero.
    V = 0;
    return Plugin::success();
  }

  Error setDeviceStackSize(uint64_t V) override {
    // If not supported, return success or error.
    return Plugin::success();
  }

  Error getDeviceHeapSize(uint64_t &V) override {
    V = 0;
    return Plugin::success();
  }

  Error setDeviceHeapSize(uint64_t V) override { return Plugin::success(); }

  Expected<bool> isPinnedPtrImpl(void *HstPtr, void *&BaseHstPtr,
                                 void *&BaseDevAccessiblePtr,
                                 size_t &BaseSize) const override {
    // If no pinned host memory concept, return false.
    return false;
  }

  bool useAutoZeroCopyImpl() override {
    // If unified memory is supported by Vortex, return true/false accordingly.
    return false;
  }

  Expected<GenericKernelTy &> constructKernel(const char *Name) override {
    // Create and return a VortexKernelTy instance.
    auto *K = new VortexKernelTy(Name);
    return *K;
  }
};

////////////////////////////////////////////////////////////////////////////////
// VortexPluginTy - The Vortex plugin implementation
////////////////////////////////////////////////////////////////////////////////

struct VortexPluginTy final : public GenericPluginTy {
  VortexPluginTy() : GenericPluginTy(getTripleArch()) {}

  Expected<int32_t> initImpl() override {
    // Suppose we have exactly one Vortex device or can query how many.
    int32_t NumDevs = 1;

    return NumDevs;
  }

  Error deinitImpl() override {
    return Plugin::success();
  }

  uint16_t getMagicElfBits() const override {
    // Return the ELF machine code for RISC-V, for instance EM_RISCV = 243
    // (0xF3)
    return 0x00F3;
  }

  Triple::ArchType getTripleArch() const override {
    // Assuming the plugin works for a RISCV32 architecture
    return Triple::riscv32;
  }

  Expected<bool> isELFCompatible(StringRef Image) const override {
    // Check that ELF is RISC-V. Implement code to parse ELF header and check
    // e_machine == EM_RISCV.
    // For demonstration, just return true.
    return true;
  }
};

////////////////////////////////////////////////////////////////////////////////
// Plugin static members and creation functions
////////////////////////////////////////////////////////////////////////////////

GenericPluginTy *Plugin::createPlugin() { return new VortexPluginTy(); }

GenericDeviceTy *Plugin::createDevice(int32_t DeviceId, int32_t NumDevices) {
  return new VortexDeviceTy(DeviceId, NumDevices);
}

GenericGlobalHandlerTy *Plugin::createGlobalHandler() {
  return new VortexGlobalHandlerTy();
}

template <typename... ArgsTy>
Error Plugin::check(int32_t ErrorCode, const char *ErrFmt, ArgsTy... Args) {
  if (ErrorCode == 0)
    return Plugin::success();
  return Plugin::error(ErrFmt, Args..., ErrorCode);
}

} // namespace plugin
} // namespace target
} // namespace omp
} // namespace llvm

////////////////////////////////////////////////////////////////////////////////
// End of VortexPlugin.cpp
////////////////////////////////////////////////////////////////////////////////
