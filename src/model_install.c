#define _POSIX_C_SOURCE 200809L
#include "stt/model_install.h"
#include "stt/fs.h"

#include <curl/curl.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
  const char *name;
  const char *url;
} ModelFile;

static const ModelFile V3_FILES[] = {
  {"config.json", "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/config.json"},
  {"generation_config.json", "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/generation_config.json"},
  {"model.safetensors", "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/model.safetensors"},
  {"processor_config.json", "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/processor_config.json"},
  {"tokenizer.json", "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/tokenizer.json"},
  {"tokenizer_config.json", "https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3/resolve/main/tokenizer_config.json"},
};

static int download_file(const char *url, const char *path) {
  CURL *curl = curl_easy_init();
  if (!curl) return -1;
  int ok = -1;
  size_t part_len = strlen(path) + 6;
  char *part = malloc(part_len);
  if (!part) goto done;
  snprintf(part, part_len, "%s.part", path);
  FILE *f = fopen(part, "wb");
  if (!f) goto done;
  curl_easy_setopt(curl, CURLOPT_URL, url);
  curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, f);
  curl_easy_setopt(curl, CURLOPT_FAILONERROR, 1L);
  curl_easy_setopt(curl, CURLOPT_USERAGENT, "stt-c/0.1");
  CURLcode rc = curl_easy_perform(curl);
  fclose(f);
  if (rc != CURLE_OK) {
    fprintf(stderr, "download failed: %s: %s\n", url, curl_easy_strerror(rc));
    unlink(part);
    goto done;
  }
  if (rename(part, path) != 0) {
    fprintf(stderr, "rename failed: %s -> %s: %s\n", part, path, strerror(errno));
    unlink(part);
    goto done;
  }
  ok = 0;

done:
  free(part);
  curl_easy_cleanup(curl);
  return ok;
}

int stt_install_model_v3(const char *model_dir_arg) {
  char *model_dir = stt_expand_home(model_dir_arg);
  int curl_started = 0;
  int ok = -1;
  if (!model_dir) return ok;
  if (stt_mkdir_p(model_dir) != 0) {
    perror(model_dir);
    goto done;
  }
  curl_global_init(CURL_GLOBAL_DEFAULT);
  curl_started = 1;
  for (size_t i = 0; i < sizeof(V3_FILES) / sizeof(V3_FILES[0]); ++i) {
    char *dst = stt_path_join(model_dir, V3_FILES[i].name);
    if (!dst) goto done;
    if (stt_file_exists(dst)) {
      fprintf(stderr, "exists: %s\n", dst);
      free(dst);
      continue;
    }
    fprintf(stderr, "downloading %s\n", V3_FILES[i].name);
    if (download_file(V3_FILES[i].url, dst) != 0) {
      free(dst);
      goto done;
    }
    free(dst);
  }
  char *source = stt_path_join(model_dir, ".source.json");
  if (source) {
    FILE *f = fopen(source, "wb");
    if (f) {
      fputs("{\"source\":\"huggingface:nvidia/parakeet-tdt-0.6b-v3\",\"revision\":\"main\"}\n", f);
      fclose(f);
    }
    free(source);
  }
  ok = 0;

done:
  if (curl_started) curl_global_cleanup();
  free(model_dir);
  return ok;
}
