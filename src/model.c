#define _POSIX_C_SOURCE 200809L
#include "stt/model.h"
#include "stt/cuda_runtime.h"
#include "stt/fs.h"
#include "stt/log.h"
#include "stt/safetensors.h"
#include "stt/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *REQUIRED_FILES[] = {
  "model.safetensors",
  "config.json",
  "generation_config.json",
  "processor_config.json",
  "tokenizer.json",
};

static const char *TDT_ONNX_FILES[] = {
  "encoder-model.onnx",
  "decoder_joint-model.onnx",
  "vocab.txt",
};

static const char *REQUIRED_TENSORS[] = {
  "decoder.embedding.weight",
  "decoder.lstm.weight_ih_l0",
  "decoder.lstm.weight_hh_l0",
  "decoder.lstm.weight_ih_l1",
  "decoder.lstm.weight_hh_l1",
  "encoder_projector.weight",
  "joint.head.weight",
  "joint.head.bias",
};

typedef struct {
  char *name;
  CUdeviceptr ptr;
  size_t bytes;
} SttGpuTensor;

struct SttModel {
  SttModelKind kind;
  char *model_dir;
  SttCuda cuda;
  SttSafeTensors tensors;
  SttGpuTensor *gpu_tensors;
  size_t gpu_tensor_count;
  SttTokenizer tokenizer;
  int loaded;
};

static int files_exist(const char *model_dir, const char *const *files, size_t count, int print_missing) {
  int ok = 0;
  for (size_t i = 0; i < count; ++i) {
    char *path = stt_path_join(model_dir, files[i]);
    if (!path || !stt_file_exists(path)) {
      if (print_missing) LOG_ERROR("missing model file: %s/%s\n", model_dir, files[i]);
      ok = -1;
    }
    free(path);
  }
  return ok;
}

static int validate_v3_tensors(const SttSafeTensors *st) {
  for (size_t i = 0; i < sizeof(REQUIRED_TENSORS) / sizeof(REQUIRED_TENSORS[0]); ++i) {
    if (!stt_safetensors_find(st, REQUIRED_TENSORS[i])) {
      LOG_ERROR("missing expected v3 tensor: %s\n", REQUIRED_TENSORS[i]);
      return -1;
    }
  }
  const SttTensorInfo *joint = stt_safetensors_find(st, "joint.head.weight");
  if (!joint || joint->rank != 2 || joint->shape[0] != 8198 || joint->shape[1] != 640) {
    LOG_ERROR("unexpected joint.head.weight shape; this does not look like Parakeet-TDT v3\n");
    return -1;
  }
  return 0;
}

int stt_model_inspect(const char *model_dir_arg) {
  char *model_dir = stt_expand_home(model_dir_arg);
  if (!model_dir) return -1;
  if (files_exist(model_dir, TDT_ONNX_FILES, sizeof(TDT_ONNX_FILES) / sizeof(TDT_ONNX_FILES[0]), 0) == 0) {
    printf("model: %s\n", model_dir);
    printf("type: TDT ONNX\n");
    printf("encoder: encoder-model.onnx\n");
    printf("decoder/joint: decoder_joint-model.onnx\n");
    printf("vocab: vocab.txt\n");
    free(model_dir);
    return 0;
  }

  char *path = stt_path_join(model_dir, "model.safetensors");
  if (!path) {
    free(model_dir);
    return -1;
  }
  SttSafeTensors st;
  if (stt_safetensors_open(&st, path) != 0) {
    free(path);
    free(model_dir);
    return -1;
  }
  size_t total = 0;
  size_t uploadable = 0;
  for (size_t i = 0; i < st.count; ++i) {
    size_t n = stt_tensor_nbytes(&st.items[i]);
    total += n;
    if (st.items[i].dtype != STT_TENSOR_I64) uploadable += n;
  }
  printf("model: %s\n", model_dir);
  printf("tensors: %zu\n", st.count);
  printf("total tensor bytes: %.2f GiB\n", (double)total / (1024.0 * 1024.0 * 1024.0));
  printf("GPU-uploadable bytes: %.2f GiB\n", (double)uploadable / (1024.0 * 1024.0 * 1024.0));
  validate_v3_tensors(&st);
  stt_safetensors_free(&st);
  free(path);
  free(model_dir);
  return 0;
}

int stt_model_load(SttModel **model_out, const char *model_dir_arg) {
  char *weights = NULL;
  char *tokenizer = NULL;
  if (!model_out) return -1;
  SttModel *model = calloc(1, sizeof(*model));
  if (!model) return -1;
  *model_out = NULL;
  model->model_dir = stt_expand_home(model_dir_arg);
  if (!model->model_dir) goto fail;

  if (files_exist(model->model_dir, TDT_ONNX_FILES, sizeof(TDT_ONNX_FILES) / sizeof(TDT_ONNX_FILES[0]), 0) == 0) {
    model->kind = STT_MODEL_TDT_ONNX;
    model->loaded = 1;
    LOG_INFO("loaded TDT ONNX model: %s\n", model->model_dir);
    *model_out = model;
    return 0;
  }

  model->kind = STT_MODEL_V3_SAFETENSORS;
  if (files_exist(model->model_dir, REQUIRED_FILES, sizeof(REQUIRED_FILES) / sizeof(REQUIRED_FILES[0]), 1) != 0) goto fail;

  weights = stt_path_join(model->model_dir, "model.safetensors");
  tokenizer = stt_path_join(model->model_dir, "tokenizer.json");
  if (!weights || !tokenizer) goto fail;
  if (stt_safetensors_open(&model->tensors, weights) != 0) goto fail;
  if (validate_v3_tensors(&model->tensors) != 0) goto fail;
  if (stt_tokenizer_load(&model->tokenizer, tokenizer) != 0) {
    LOG_ERROR("failed to load tokenizer: %s\n", tokenizer);
    goto fail;
  }
  if (stt_cuda_init(&model->cuda) != 0) {
    LOG_ERROR("CUDA unavailable: %s\n", stt_cuda_last_error());
    goto fail;
  }

  model->gpu_tensors = calloc(model->tensors.count, sizeof(*model->gpu_tensors));
  if (!model->gpu_tensors) goto fail;
  for (size_t i = 0; i < model->tensors.count; ++i) {
    const SttTensorInfo *info = &model->tensors.items[i];
    if (info->dtype == STT_TENSOR_I64) continue;
    void *host = NULL;
    size_t bytes = 0;
    if (stt_safetensors_read_tensor(&model->tensors, info, &host, &bytes) != 0) {
      LOG_ERROR("failed reading tensor: %s\n", info->name);
      goto fail;
    }
    CUdeviceptr ptr = 0;
    if (stt_cuda_upload(host, bytes, &ptr) != 0) {
      LOG_ERROR("failed uploading tensor %s: %s\n", info->name, stt_cuda_last_error());
      free(host);
      goto fail;
    }
    free(host);
    model->gpu_tensors[model->gpu_tensor_count].name = strdup(info->name);
    if (!model->gpu_tensors[model->gpu_tensor_count].name) {
      cuMemFree(ptr);
      goto fail;
    }
    model->gpu_tensors[model->gpu_tensor_count].ptr = ptr;
    model->gpu_tensors[model->gpu_tensor_count].bytes = bytes;
    model->gpu_tensor_count++;
  }
  model->loaded = 1;
  LOG_INFO("loaded %zu tensors to %s VRAM\n", model->gpu_tensor_count, model->cuda.name);
  free(weights);
  free(tokenizer);
  *model_out = model;
  return 0;

fail:
  free(weights);
  free(tokenizer);
  stt_model_free(model);
  return -1;
}

void stt_model_free(SttModel *model) {
  if (!model) return;
  for (size_t i = 0; i < model->gpu_tensor_count; ++i) {
    if (model->gpu_tensors[i].ptr) cuMemFree(model->gpu_tensors[i].ptr);
    free(model->gpu_tensors[i].name);
  }
  free(model->gpu_tensors);
  stt_tokenizer_free(&model->tokenizer);
  stt_safetensors_free(&model->tensors);
  stt_cuda_destroy(&model->cuda);
  free(model->model_dir);
  free(model);
}

SttModelKind stt_model_kind(const SttModel *model) {
  return model ? model->kind : STT_MODEL_UNKNOWN;
}

const char *stt_model_dir(const SttModel *model) {
  return model ? model->model_dir : NULL;
}
