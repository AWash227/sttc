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
#if STT_ENABLE_DOWNLOAD
#include <winhttp.h>
#endif
#endif

#if STT_ENABLE_DOWNLOAD && !defined(_WIN32)
#include <curl/curl.h>
#endif

typedef struct {
  const char *name;
} ModelFile;

static const char *MODEL_REVISION = "8f23f0c03c8761650bdb5b40aaf3e40d2c15f1ce";
#if STT_ENABLE_DOWNLOAD
static const char *MODEL_REPO = "istupakov/parakeet-tdt-0.6b-v3-onnx";
#endif
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
  return "Ctrl+Shift";
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

#if STT_ENABLE_DOWNLOAD && defined(_WIN32)
static wchar_t *utf8_to_wide(const char *text) {
  int needed = MultiByteToWideChar(CP_UTF8, 0, text, -1, NULL, 0);
  if (needed <= 0) return NULL;
  wchar_t *wide = malloc((size_t)needed * sizeof(*wide));
  if (!wide) return NULL;
  if (MultiByteToWideChar(CP_UTF8, 0, text, -1, wide, needed) <= 0) {
    free(wide);
    return NULL;
  }
  return wide;
}

static int download_url_to_file(const char *url, const char *path) {
  int rc = -1;
  wchar_t *url_w = utf8_to_wide(url);
  URL_COMPONENTS parts;
  wchar_t host[256];
  wchar_t path_buf[2048];
  wchar_t extra_buf[2048];
  HINTERNET session = NULL;
  HINTERNET connect = NULL;
  HINTERNET request = NULL;
  FILE *f = NULL;

  if (!url_w) return -1;
  memset(&parts, 0, sizeof(parts));
  parts.dwStructSize = sizeof(parts);
  parts.lpszHostName = host;
  parts.dwHostNameLength = sizeof(host) / sizeof(host[0]);
  parts.lpszUrlPath = path_buf;
  parts.dwUrlPathLength = sizeof(path_buf) / sizeof(path_buf[0]);
  parts.lpszExtraInfo = extra_buf;
  parts.dwExtraInfoLength = sizeof(extra_buf) / sizeof(extra_buf[0]);

  if (!WinHttpCrackUrl(url_w, 0, 0, &parts)) {
    LOG_ERROR("model download failed to parse URL: %s error=%lu\n", url, GetLastError());
    goto done;
  }

  size_t target_len = (size_t)parts.dwUrlPathLength + (size_t)parts.dwExtraInfoLength;
  wchar_t *target = malloc((target_len + 1) * sizeof(*target));
  if (!target) goto done;
  memcpy(target, parts.lpszUrlPath, (size_t)parts.dwUrlPathLength * sizeof(*target));
  memcpy(target + parts.dwUrlPathLength, parts.lpszExtraInfo, (size_t)parts.dwExtraInfoLength * sizeof(*target));
  target[target_len] = L'\0';

  session = WinHttpOpen(L"stt/1.0", WINHTTP_ACCESS_TYPE_DEFAULT_PROXY, WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS, 0);
  if (!session) {
    LOG_ERROR("model download failed to open WinHTTP session: error=%lu\n", GetLastError());
    free(target);
    goto done;
  }
  DWORD redirects = WINHTTP_OPTION_REDIRECT_POLICY_ALWAYS;
  WinHttpSetOption(session, WINHTTP_OPTION_REDIRECT_POLICY, &redirects, sizeof(redirects));

  connect = WinHttpConnect(session, parts.lpszHostName, parts.nPort, 0);
  if (!connect) {
    LOG_ERROR("model download failed to connect: %s error=%lu\n", url, GetLastError());
    free(target);
    goto done;
  }

  request = WinHttpOpenRequest(connect,
                               L"GET",
                               target,
                               NULL,
                               WINHTTP_NO_REFERER,
                               WINHTTP_DEFAULT_ACCEPT_TYPES,
                               parts.nScheme == INTERNET_SCHEME_HTTPS ? WINHTTP_FLAG_SECURE : 0);
  free(target);
  if (!request) {
    LOG_ERROR("model download failed to create request: %s error=%lu\n", url, GetLastError());
    goto done;
  }

  if (!WinHttpSendRequest(request, WINHTTP_NO_ADDITIONAL_HEADERS, 0, WINHTTP_NO_REQUEST_DATA, 0, 0, 0) ||
      !WinHttpReceiveResponse(request, NULL)) {
    LOG_ERROR("model download request failed: %s error=%lu\n", url, GetLastError());
    goto done;
  }

  DWORD status = 0;
  DWORD status_len = sizeof(status);
  if (WinHttpQueryHeaders(request,
                          WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                          WINHTTP_HEADER_NAME_BY_INDEX,
                          &status,
                          &status_len,
                          WINHTTP_NO_HEADER_INDEX) &&
      (status < 200 || status >= 300)) {
    LOG_ERROR("model download returned HTTP status %lu: %s\n", status, url);
    goto done;
  }

  f = fopen(path, "wb");
  if (!f) goto done;
  for (;;) {
    DWORD available = 0;
    if (!WinHttpQueryDataAvailable(request, &available)) {
      LOG_ERROR("model download failed while reading: %s error=%lu\n", url, GetLastError());
      goto done;
    }
    if (available == 0) break;
    char buf[65536];
    while (available > 0) {
      DWORD chunk = available > sizeof(buf) ? sizeof(buf) : available;
      DWORD read = 0;
      if (!WinHttpReadData(request, buf, chunk, &read)) {
        LOG_ERROR("model download failed while streaming: %s error=%lu\n", url, GetLastError());
        goto done;
      }
      if (read == 0) break;
      if (fwrite(buf, 1, read, f) != read) goto done;
      available -= read;
    }
  }
  rc = 0;

done:
  if (f && fclose(f) != 0) rc = -1;
  if (request) WinHttpCloseHandle(request);
  if (connect) WinHttpCloseHandle(connect);
  if (session) WinHttpCloseHandle(session);
  free(url_w);
  return rc;
}

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
  if (download_url_to_file(url, partial) != 0 || rename(partial, path) != 0) {
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
  for (size_t i = 0; i < sizeof(MODEL_FILES) / sizeof(MODEL_FILES[0]); ++i) {
    char *path = stt_path_join(model_dir, MODEL_FILES[i].name);
    int exists = path && stt_file_exists(path);
    free(path);
    if (exists) {
      LOG_INFO("Model file already exists: %s\n", MODEL_FILES[i].name);
      continue;
    }
    LOG_INFO("Downloading model file: %s\n", MODEL_FILES[i].name);
    if (download_one(model_dir, MODEL_FILES[i].name) != 0) return -1;
  }
  return 0;
}
#elif STT_ENABLE_DOWNLOAD
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
    char *path = stt_path_join(model_dir, MODEL_FILES[i].name);
    int exists = path && stt_file_exists(path);
    free(path);
    if (exists) {
      LOG_INFO("Model file already exists: %s\n", MODEL_FILES[i].name);
      continue;
    }
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
