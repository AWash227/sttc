#define _POSIX_C_SOURCE 200809L
#include "stt/model.h"
#include "stt/fs.h"
#include "stt/log.h"

#include <stdlib.h>

static const char *TDT_ONNX_FILES[] = {
  "encoder-model.onnx",
  "encoder-model.onnx.data",
  "decoder_joint-model.onnx",
  "config.json",
  "vocab.txt",
};

struct SttModel {
  char *model_dir;
};

static int tdt_files_exist(const char *model_dir) {
  for (size_t i = 0; i < sizeof(TDT_ONNX_FILES) / sizeof(TDT_ONNX_FILES[0]); ++i) {
    char *path = stt_path_join(model_dir, TDT_ONNX_FILES[i]);
    int exists = path && stt_file_exists(path);
    free(path);
    if (!exists) {
      LOG_ERROR("missing model file: %s/%s\n", model_dir, TDT_ONNX_FILES[i]);
      return 0;
    }
  }
  return 1;
}

int stt_model_load(SttModel **model_out, const char *model_dir_arg) {
  if (!model_out) return -1;
  *model_out = NULL;

  SttModel *model = calloc(1, sizeof(*model));
  if (!model) return -1;

  model->model_dir = stt_expand_home(model_dir_arg);
  if (!model->model_dir || !tdt_files_exist(model->model_dir)) {
    stt_model_free(model);
    return -1;
  }

  LOG_INFO("Loaded speech model: %s\n", model->model_dir);
  *model_out = model;
  return 0;
}

void stt_model_free(SttModel *model) {
  if (!model) return;
  free(model->model_dir);
  free(model);
}

const char *stt_model_dir(const SttModel *model) {
  return model ? model->model_dir : NULL;
}
