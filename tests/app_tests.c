#define _POSIX_C_SOURCE 200809L
#include "stt/app.h"
#include "stt/fs.h"

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <direct.h>
static void set_env_value(const char *key, const char *value) {
  _putenv_s(key, value);
}
#else
static void set_env_value(const char *key, const char *value) {
  setenv(key, value, 1);
}
#endif

static void touch_file(const char *dir, const char *name) {
  char *path = stt_path_join(dir, name);
  assert(path);
  FILE *f = fopen(path, "wb");
  assert(f);
  fclose(f);
  free(path);
}

static char *test_root(void) {
  const char *tmp = getenv("TMPDIR");
  if (!tmp) tmp = getenv("TEMP");
  if (!tmp) tmp = "/tmp";
  char suffix[64];
  snprintf(suffix, sizeof(suffix), "stt-app-test-%ld", (long)time(NULL));
  char *root = stt_path_join(tmp, suffix);
  assert(root);
  assert(stt_mkdir_p(root) == 0);
  return root;
}

static void create_model_bundle(const char *model_dir) {
  assert(stt_mkdir_p(model_dir) == 0);
  touch_file(model_dir, "encoder-model.onnx");
  touch_file(model_dir, "encoder-model.onnx.data");
  touch_file(model_dir, "decoder_joint-model.onnx");
  touch_file(model_dir, "config.json");
  touch_file(model_dir, "vocab.txt");
}

static void test_prepare_creates_config_and_uses_default_model_dir(void) {
  char *root = test_root();
  char *config_home = stt_path_join(root, "config-home");
  char *home = stt_path_join(root, "home");
  assert(config_home && home);
  assert(stt_mkdir_p(config_home) == 0);
  assert(stt_mkdir_p(home) == 0);

  char *model_dir = NULL;
#ifdef _WIN32
  model_dir = stt_path_join(home, "parakeet-tdt");
#else
  set_env_value("XDG_CONFIG_HOME", config_home);
  set_env_value("HOME", home);
  char *models = stt_path_join(home, ".models");
  model_dir = stt_path_join(models, "parakeet-tdt");
  free(models);
#endif
  set_env_value("STT_CONFIG_HOME", config_home);
  set_env_value("STT_MODEL_DIR", model_dir);
  create_model_bundle(model_dir);

  SttConfig config = {0};
  config.infer_provider = "auto";
  config.model_variant = "auto";
  assert(stt_app_prepare(&config) == 0);
  assert(config.model_dir);
  assert(strcmp(config.model_dir, model_dir) == 0);

  char *config_dir = strdup(config_home);
  char *config_path = stt_path_join(config_dir, "config.toml");
  assert(config_path && stt_file_exists(config_path));

  free((char *)config.model_dir);
  free(config_path);
  free(config_dir);
  free(model_dir);
  free(config_home);
  free(home);
  free(root);
}

static void test_prepare_declines_missing_model(void) {
  char *root = test_root();
  char *config_home = stt_path_join(root, "missing-config-home");
  char *home = stt_path_join(root, "missing-home");
  assert(config_home && home);
  assert(stt_mkdir_p(config_home) == 0);
  assert(stt_mkdir_p(home) == 0);
  set_env_value("STT_ASSUME_NO_DOWNLOAD", "1");
#ifdef _WIN32
  set_env_value("APPDATA", config_home);
  set_env_value("LOCALAPPDATA", home);
#else
  set_env_value("XDG_CONFIG_HOME", config_home);
  set_env_value("HOME", home);
#endif
  char *missing_model = stt_path_join(home, "missing-model");
  set_env_value("STT_CONFIG_HOME", config_home);
  set_env_value("STT_MODEL_DIR", missing_model);
  SttConfig config = {0};
  config.infer_provider = "auto";
  config.model_variant = "auto";
  assert(stt_app_prepare(&config) != 0);
  set_env_value("STT_ASSUME_NO_DOWNLOAD", "");

  free(config_home);
  free(home);
  free(missing_model);
  free(root);
}

int main(void) {
  test_prepare_creates_config_and_uses_default_model_dir();
  test_prepare_declines_missing_model();
  return 0;
}
