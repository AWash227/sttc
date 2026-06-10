#ifndef STT_CUDA_RUNTIME_H
#define STT_CUDA_RUNTIME_H

#include <stddef.h>
#include <cuda.h>

typedef struct {
  CUdevice device;
  CUcontext context;
  char name[256];
  size_t total_mem;
} SttCuda;

int stt_cuda_init(SttCuda *cuda);
void stt_cuda_destroy(SttCuda *cuda);
int stt_cuda_upload(const void *host, size_t bytes, CUdeviceptr *device_ptr);
const char *stt_cuda_last_error(void);

#endif
