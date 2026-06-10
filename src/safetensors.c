#define _POSIX_C_SOURCE 200809L
#include "stt/safetensors.h"
#include "stt/fs.h"

#include <json-c/json.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t read_le64(const unsigned char b[8]) {
  uint64_t v = 0;
  for (int i = 7; i >= 0; --i) v = (v << 8) | b[i];
  return v;
}

static SttTensorDType parse_dtype(const char *s) {
  if (strcmp(s, "F32") == 0) return STT_TENSOR_F32;
  if (strcmp(s, "F16") == 0) return STT_TENSOR_F16;
  if (strcmp(s, "BF16") == 0) return STT_TENSOR_BF16;
  if (strcmp(s, "I64") == 0) return STT_TENSOR_I64;
  return STT_TENSOR_UNKNOWN;
}

const char *stt_tensor_dtype_name(SttTensorDType dtype) {
  switch (dtype) {
    case STT_TENSOR_F32: return "F32";
    case STT_TENSOR_F16: return "F16";
    case STT_TENSOR_BF16: return "BF16";
    case STT_TENSOR_I64: return "I64";
    default: return "UNKNOWN";
  }
}

size_t stt_tensor_dtype_size(SttTensorDType dtype) {
  switch (dtype) {
    case STT_TENSOR_F32: return 4;
    case STT_TENSOR_F16:
    case STT_TENSOR_BF16: return 2;
    case STT_TENSOR_I64: return 8;
    default: return 0;
  }
}

size_t stt_tensor_nbytes(const SttTensorInfo *info) {
  size_t n = stt_tensor_dtype_size(info->dtype);
  for (size_t i = 0; i < info->rank; ++i) n *= info->shape[i];
  return n;
}

int stt_safetensors_open(SttSafeTensors *st, const char *path) {
  memset(st, 0, sizeof(*st));
  FILE *f = fopen(path, "rb");
  if (!f) {
    perror(path);
    return -1;
  }
  unsigned char len_buf[8];
  if (fread(len_buf, 1, 8, f) != 8) {
    fclose(f);
    return -1;
  }
  uint64_t header_len = read_le64(len_buf);
  if (header_len == 0 || header_len > 16 * 1024 * 1024) {
    fclose(f);
    fprintf(stderr, "invalid safetensors header length: %llu\n", (unsigned long long)header_len);
    return -1;
  }
  char *header = malloc((size_t)header_len + 1);
  if (!header) {
    fclose(f);
    return -1;
  }
  if (fread(header, 1, (size_t)header_len, f) != (size_t)header_len) {
    free(header);
    fclose(f);
    return -1;
  }
  fclose(f);
  header[header_len] = '\0';

  json_object *root = json_tokener_parse(header);
  free(header);
  if (!root || !json_object_is_type(root, json_type_object)) {
    fprintf(stderr, "failed to parse safetensors JSON header\n");
    if (root) json_object_put(root);
    return -1;
  }

  st->path = strdup(path);
  st->data_start = 8 + header_len;

  json_object_object_foreach(root, key, val) {
    if (strcmp(key, "__metadata__") == 0) continue;
    json_object *dtype_o = NULL, *shape_o = NULL, *offsets_o = NULL;
    if (!json_object_object_get_ex(val, "dtype", &dtype_o) ||
        !json_object_object_get_ex(val, "shape", &shape_o) ||
        !json_object_object_get_ex(val, "data_offsets", &offsets_o)) {
      continue;
    }
    size_t idx = st->count++;
    SttTensorInfo *next = realloc(st->items, st->count * sizeof(*st->items));
    if (!next) {
      json_object_put(root);
      return -1;
    }
    st->items = next;
    SttTensorInfo *info = &st->items[idx];
    memset(info, 0, sizeof(*info));
    info->name = strdup(key);
    info->dtype = parse_dtype(json_object_get_string(dtype_o));
    info->rank = (size_t)json_object_array_length(shape_o);
    info->shape = calloc(info->rank ? info->rank : 1, sizeof(size_t));
    if (!info->name || !info->shape) {
      json_object_put(root);
      return -1;
    }
    for (size_t i = 0; i < info->rank; ++i) {
      info->shape[i] = (size_t)json_object_get_int64(json_object_array_get_idx(shape_o, i));
    }
    info->data_begin = (uint64_t)json_object_get_int64(json_object_array_get_idx(offsets_o, 0));
    info->data_end = (uint64_t)json_object_get_int64(json_object_array_get_idx(offsets_o, 1));
  }

  json_object_put(root);
  return 0;
}

void stt_safetensors_free(SttSafeTensors *st) {
  if (!st) return;
  for (size_t i = 0; i < st->count; ++i) {
    free(st->items[i].name);
    free(st->items[i].shape);
  }
  free(st->items);
  free(st->path);
  memset(st, 0, sizeof(*st));
}

const SttTensorInfo *stt_safetensors_find(const SttSafeTensors *st, const char *name) {
  for (size_t i = 0; i < st->count; ++i) {
    if (strcmp(st->items[i].name, name) == 0) return &st->items[i];
  }
  return NULL;
}

int stt_safetensors_read_tensor(const SttSafeTensors *st, const SttTensorInfo *info, void **data_out, size_t *bytes_out) {
  uint64_t bytes64 = info->data_end - info->data_begin;
  size_t bytes = (size_t)bytes64;
  void *buf = malloc(bytes);
  if (!buf) return -1;
  FILE *f = fopen(st->path, "rb");
  if (!f) {
    free(buf);
    return -1;
  }
  if (fseeko(f, (off_t)(st->data_start + info->data_begin), SEEK_SET) != 0) {
    fclose(f);
    free(buf);
    return -1;
  }
  size_t got = fread(buf, 1, bytes, f);
  fclose(f);
  if (got != bytes) {
    free(buf);
    return -1;
  }
  *data_out = buf;
  *bytes_out = bytes;
  return 0;
}
