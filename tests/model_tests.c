#define _POSIX_C_SOURCE 200809L
#include "stt/model.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <direct.h>
#include <io.h>
#include <windows.h>
#define stt_rmdir _rmdir
#define stt_unlink _unlink
#else
#include <unistd.h>
#define stt_rmdir rmdir
#define stt_unlink unlink
#endif

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
    stt_unlink(path);
  }
  stt_rmdir(dir);
}

static char *make_temp_dir(void) {
#ifdef _WIN32
  const char *base = getenv("TEMP");
  if (!base || !*base) base = getenv("TMP");
  if (!base || !*base) base = ".";
  char path[512];
  for (int i = 0; i < 100; ++i) {
    snprintf(path, sizeof(path), "%s\\stt-model-test-%lu-%d", base, (unsigned long)GetCurrentProcessId(), i);
    if (_mkdir(path) == 0) return strdup(path);
  }
  return NULL;
#else
  char tmpl[] = "/tmp/stt-model-test-XXXXXX";
  char *dir = mkdtemp(tmpl);
  return dir ? strdup(dir) : NULL;
#endif
}

static void test_requires_full_bundle(void) {
  char *dir = make_temp_dir();
  assert(dir);

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
  free(dir);
}

int main(void) {
  test_requires_full_bundle();
  return 0;
}
