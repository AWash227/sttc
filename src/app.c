#define _POSIX_C_SOURCE 200809L
#include "stt/app.h"
#include "stt/fs.h"
#include "stt/log.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <shlobj.h>
#endif

#if STT_ENABLE_DOWNLOAD
#include <curl/curl.h>
#endif

typedef struct {
  const char *name;
} ModelFile;

#if STT_ENABLE_DOWNLOAD
static const char *MODEL_REPO = "onnx-asr/nemo-parakeet-tdt-0.6b-v2";
#endif
static const char *MODEL_REVISION = "d808c3be882f47cf6a15a42c0eb9ee751b99a379";
static const ModelFile MODEL_FILES[] = {
  {"encoder-model.onnx"},
  {"encoder-model.onnx.data"},
  {"decoder_joint-model.onnx"},
  {"config.json"},
  {"vocab.txt"},
};

static char *stt_strdup_or_null(const char *s) {
  return s ? strdup(s) : NULL;
}

static char *path_from_env(const char *env_name, const char *fallback) {
  const char *value = getenv(env_name);
  if (value && *value) return strdup(value);
  return fallback ? stt_expand_home(fallback) : NULL;
}

#ifdef _WIN32
static char *known_folder(REFKNOWNFOLDERID id, const char *fallback_env) {
  PWSTR wide = NULL;
  if (SUCCEEDED(SHGetKnownFolderPath(id, 0, NULL, &wide))) {
    int needed = WideCharToMultiByte(CP_UTF8, 0, wide, -1, NULL, 0, NULL, NULL);
    char *out = needed > 0 ? malloc((size_t)needed) : NULL;
    if (out) WideCharToMultiByte(CP_UTF8, 0, wide, -1, out, needed, NULL, NULL);
    CoTaskMemFree(wide);
    if (out) return out;
  }
  return path_from_env(fallback_env, NULL);
}
#endif

static char *default_config_dir(void) {
  char *override = path_from_env("STT_CONFIG_HOME", NULL);
  if (override) return override;
#ifdef _WIN32
  char *base = known_folder(&FOLDERID_RoamingAppData, "APPDATA");
  if (!base) return NULL;
  char *dir = stt_path_join(base, "stt");
  free(base);
  return dir;
#else
  char *base = path_from_env("XDG_CONFIG_HOME", "~/.config");
  if (!base) return NULL;
  char *dir = stt_path_join(base, "stt");
  free(base);
  return dir;
#endif
}

static char *default_model_dir(void) {
  char *override = path_from_env("STT_MODEL_DIR", NULL);
  if (override) return override;
#ifdef _WIN32
  char *base = known_folder(&FOLDERID_LocalAppData, "LOCALAPPDATA");
  if (!base) return NULL;
  char *app = stt_path_join(base, "stt");
  char *models = app ? stt_path_join(app, "models") : NULL;
  char *dir = models ? stt_path_join(models, "parakeet-tdt") : NULL;
  free(base);
  free(app);
  free(models);
  return dir;
#else
  return stt_expand_home("~/.models/parakeet-tdt");
#endif
}

static char *config_path_from_dir(const char *dir) {
  return stt_path_join(dir, "config.toml");
}

static const char *default_hotkey(void) {
#ifdef _WIN32
  return "Alt+V";
#else
  return "Meta+V";
#endif
}

static int parse_string_key(const char *text, const char *key, char **out) {
  const char *p = strstr(text, key);
  if (!p) return 0;
  p += strlen(key);
  while (*p && isspace((unsigned char)*p)) p++;
  if (*p != '=') return 0;
  p++;
  while (*p && isspace((unsigned char)*p)) p++;
  if (*p != '"') return 0;
  p++;
  const char *end = strchr(p, '"');
  if (!end) return 0;
  size_t len = (size_t)(end - p);
  char *value = malloc(len + 1);
  if (!value) return -1;
  memcpy(value, p, len);
  value[len] = '\0';
  free(*out);
  *out = value;
  return 1;
}

static int write_default_config(const char *path, const char *model_dir) {
  char buf[8192];
  snprintf(buf,
           sizeof(buf),
           "model_dir = \"%s\"\n"
           "model_revision = \"%s\"\n"
           "infer_provider = \"auto\"\n"
           "model_variant = \"auto\"\n"
           "hotkey = \"%s\"\n"
           "type_delay_ms = 0\n",
           model_dir,
           MODEL_REVISION,
           default_hotkey());
  return stt_write_file(path, buf);
}

static int model_files_exist(const char *model_dir) {
  for (size_t i = 0; i < sizeof(MODEL_FILES) / sizeof(MODEL_FILES[0]); ++i) {
    char *path = stt_path_join(model_dir, MODEL_FILES[i].name);
    int exists = path && stt_file_exists(path);
    free(path);
    if (!exists) return 0;
  }
  return 1;
}

static int ask_yes_no(const char *prompt) {
  const char *assume = getenv("STT_ASSUME_YES");
  if (assume && strcmp(assume, "1") == 0) return 1;
  assume = getenv("STT_ASSUME_NO_DOWNLOAD");
  if (assume && strcmp(assume, "1") == 0) return 0;
  fprintf(stderr, "%s [y/N] ", prompt);
  fflush(stderr);
  char answer[16];
  if (!fgets(answer, sizeof(answer), stdin)) return 0;
  return answer[0] == 'y' || answer[0] == 'Y';
}

#if STT_ENABLE_DOWNLOAD
static int download_one(const char *model_dir, const char *name) {
  char url[1024];
  snprintf(url, sizeof(url), "https://huggingface.co/%s/resolve/%s/%s", MODEL_REPO, MODEL_REVISION, name);
  char *path = stt_path_join(model_dir, name);
  if (!path) return -1;
  char *partial = malloc(strlen(path) + 9);
  if (!partial) {
    free(path);
    return -1;
  }
  sprintf(partial, "%s.partial", path);
  FILE *f = fopen(partial, "wb");
  if (!f) {
    free(path);
    free(partial);
    return -1;
  }
  CURL *curl = curl_easy_init();
  if (!curl) {
    fclose(f);
    free(path);
    free(partial);
    return -1;
  }
  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, f);
  CURLcode rc = curl_easy_perform(curl);
  curl_easy_cleanup(curl);
  int close_rc = fclose(f);
  if (rc != CURLE_OK || close_rc != 0 || rename(partial, path) != 0) {
    LOG_ERROR("model download failed: %s\n", url);
    remove(partial);
    free(path);
    free(partial);
    return -1;
  }
  free(path);
  free(partial);
  return 0;
}

static int download_model(const char *model_dir) {
  if (curl_global_init(CURL_GLOBAL_DEFAULT) != 0) return -1;
  for (size_t i = 0; i < sizeof(MODEL_FILES) / sizeof(MODEL_FILES[0]); ++i) {
    LOG_INFO("Downloading model file: %s\n", MODEL_FILES[i].name);
    if (download_one(model_dir, MODEL_FILES[i].name) != 0) {
      curl_global_cleanup();
      return -1;
    }
  }
  curl_global_cleanup();
  return 0;
}
#else
static int download_model(const char *model_dir) {
  (void)model_dir;
  LOG_ERROR("model download is disabled in this build\n");
  return -1;
}
#endif

int stt_app_prepare(SttConfig *config) {
  char *config_dir = default_config_dir();
  char *model_dir = config->model_dir ? stt_strdup_or_null(config->model_dir) : default_model_dir();
  if (!config_dir || !model_dir) goto fail;
  if (stt_mkdir_p(config_dir) != 0 || stt_mkdir_p(model_dir) != 0) goto fail;

  char *config_path = config_path_from_dir(config_dir);
  if (!config_path) goto fail;
  if (stt_file_exists(config_path)) {
    size_t len = 0;
    char *text = stt_read_file(config_path, &len);
    (void)len;
    if (text && !config->model_dir) parse_string_key(text, "model_dir", &model_dir);
    free(text);
  } else if (write_default_config(config_path, model_dir) == 0) {
    LOG_INFO("Created config: %s\n", config_path);
  }

  if (!model_files_exist(model_dir)) {
    LOG_WARN("Speech model is missing: %s\n", model_dir);
    if (!ask_yes_no("Download the speech model now?")) {
      LOG_ERROR("model is required before dictation can start\n");
      free(config_path);
      goto fail;
    }
    if (download_model(model_dir) != 0 || !model_files_exist(model_dir)) {
      free(config_path);
      goto fail;
    }
  }

  config->model_dir = model_dir;
  free(config_dir);
  free(config_path);
  return 0;

fail:
  free(config_dir);
  free(model_dir);
  return -1;
}
