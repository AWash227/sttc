#ifndef STT_SAFETENSORS_H
#define STT_SAFETENSORS_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
  STT_TENSOR_UNKNOWN = 0,
  STT_TENSOR_F32,
  STT_TENSOR_F16,
  STT_TENSOR_BF16,
  STT_TENSOR_I64,
} SttTensorDType;

typedef struct {
  char *name;
  SttTensorDType dtype;
  size_t *shape;
  size_t rank;
  uint64_t data_begin;
  uint64_t data_end;
} SttTensorInfo;

typedef struct {
  char *path;
  SttTensorInfo *items;
  size_t count;
  uint64_t data_start;
} SttSafeTensors;

int stt_safetensors_open(SttSafeTensors *st, const char *path);
void stt_safetensors_free(SttSafeTensors *st);
const SttTensorInfo *stt_safetensors_find(const SttSafeTensors *st, const char *name);
const char *stt_tensor_dtype_name(SttTensorDType dtype);
size_t stt_tensor_dtype_size(SttTensorDType dtype);
size_t stt_tensor_nbytes(const SttTensorInfo *info);
int stt_safetensors_read_tensor(const SttSafeTensors *st, const SttTensorInfo *info, void **data_out, size_t *bytes_out);

#endif
