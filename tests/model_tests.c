#define _POSIX_C_SOURCE 200809L
#include "stt/model.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void touch_file(const char *dir, const char *name) {
  char path[512];
  snprintf(path, sizeof(path), "%s/%s", dir, name);
  FILE *f = fopen(path, "wb");
  assert(f);
  fclose(f);
}

static void remove_bundle(const char *dir) {
  const char *files[] = {
    "encoder-model.onnx",
    "encoder-model.onnx.data",
    "decoder_joint-model.onnx",
    "config.json",
    "vocab.txt",
  };
  char path[512];
  for (size_t i = 0; i < sizeof(files) / sizeof(files[0]); ++i) {
    snprintf(path, sizeof(path), "%s/%s", dir, files[i]);
    unlink(path);
  }
  rmdir(dir);
}

static void test_requires_full_bundle(void) {
  char dir[] = "/tmp/stt-model-test-XXXXXX";
  assert(mkdtemp(dir));

  SttModel *model = NULL;
  assert(stt_model_load(&model, dir) != 0);

  touch_file(dir, "encoder-model.onnx");
  touch_file(dir, "encoder-model.onnx.data");
  touch_file(dir, "decoder_joint-model.onnx");
  touch_file(dir, "config.json");
  touch_file(dir, "vocab.txt");
  assert(stt_model_load(&model, dir) == 0);
  assert(strcmp(stt_model_dir(model), dir) == 0);
  stt_model_free(model);

  remove_bundle(dir);
}

int main(void) {
  test_requires_full_bundle();
  return 0;
}
