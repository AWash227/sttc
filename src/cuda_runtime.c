#define _POSIX_C_SOURCE 200809L
#include "stt/cuda_runtime.h"

#include <stdio.h>
#include <string.h>

static char g_last_error[512];

static int fail_cu(CUresult r, const char *what) {
  const char *name = NULL;
  const char *msg = NULL;
  cuGetErrorName(r, &name);
  cuGetErrorString(r, &msg);
  snprintf(g_last_error, sizeof(g_last_error), "%s: %s (%s)", what, msg ? msg : "unknown", name ? name : "CUDA_ERROR");
  return -1;
}

const char *stt_cuda_last_error(void) {
  return g_last_error[0] ? g_last_error : "no CUDA error";
}

int stt_cuda_init(SttCuda *cuda) {
  memset(cuda, 0, sizeof(*cuda));
  CUresult r = cuInit(0);
  if (r != CUDA_SUCCESS) return fail_cu(r, "cuInit");
  int count = 0;
  r = cuDeviceGetCount(&count);
  if (r != CUDA_SUCCESS) return fail_cu(r, "cuDeviceGetCount");
  if (count <= 0) {
    snprintf(g_last_error, sizeof(g_last_error), "no CUDA devices found");
    return -1;
  }
  r = cuDeviceGet(&cuda->device, 0);
  if (r != CUDA_SUCCESS) return fail_cu(r, "cuDeviceGet");
  r = cuDeviceGetName(cuda->name, (int)sizeof(cuda->name), cuda->device);
  if (r != CUDA_SUCCESS) return fail_cu(r, "cuDeviceGetName");
  r = cuDeviceTotalMem(&cuda->total_mem, cuda->device);
  if (r != CUDA_SUCCESS) return fail_cu(r, "cuDeviceTotalMem");
  r = cuCtxCreate(&cuda->context, NULL, 0, cuda->device);
  if (r != CUDA_SUCCESS) return fail_cu(r, "cuCtxCreate");
  return 0;
}

void stt_cuda_destroy(SttCuda *cuda) {
  if (cuda && cuda->context) {
    cuCtxDestroy(cuda->context);
    cuda->context = NULL;
  }
}

int stt_cuda_upload(const void *host, size_t bytes, CUdeviceptr *device_ptr) {
  CUdeviceptr ptr = 0;
  CUresult r = cuMemAlloc(&ptr, bytes);
  if (r != CUDA_SUCCESS) return fail_cu(r, "cuMemAlloc");
  r = cuMemcpyHtoD(ptr, host, bytes);
  if (r != CUDA_SUCCESS) {
    cuMemFree(ptr);
    return fail_cu(r, "cuMemcpyHtoD");
  }
  *device_ptr = ptr;
  return 0;
}
