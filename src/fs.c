#define _POSIX_C_SOURCE 200809L
#include "stt/fs.h"

#include <errno.h>
#include <pwd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

char *stt_expand_home(const char *path) {
  if (!path) return NULL;
  if (path[0] != '~') return strdup(path);
  if (path[1] != '/' && path[1] != '\0') return strdup(path);

  const char *home = getenv("HOME");
  if (!home) {
    struct passwd *pw = getpwuid(getuid());
    if (pw) home = pw->pw_dir;
  }
  if (!home) return NULL;

  size_t home_len = strlen(home);
  size_t rest_len = strlen(path + 1);
  char *out = malloc(home_len + rest_len + 1);
  if (!out) return NULL;
  memcpy(out, home, home_len);
  memcpy(out + home_len, path + 1, rest_len + 1);
  return out;
}

char *stt_path_join(const char *a, const char *b) {
  size_t alen = strlen(a);
  size_t blen = strlen(b);
  int slash = alen > 0 && a[alen - 1] == '/';
  char *out = malloc(alen + blen + (slash ? 1 : 2));
  if (!out) return NULL;
  memcpy(out, a, alen);
  if (!slash) out[alen++] = '/';
  memcpy(out + alen, b, blen + 1);
  return out;
}

int stt_mkdir_p(const char *path) {
  char *tmp = strdup(path);
  if (!tmp) return -1;
  for (char *p = tmp + 1; *p; ++p) {
    if (*p == '/') {
      *p = '\0';
      if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        free(tmp);
        return -1;
      }
      *p = '/';
    }
  }
  if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
    free(tmp);
    return -1;
  }
  free(tmp);
  return 0;
}

int stt_file_exists(const char *path) {
  struct stat st;
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

char *stt_read_file(const char *path, size_t *len_out) {
  FILE *f = fopen(path, "rb");
  if (!f) return NULL;
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return NULL;
  }
  long n = ftell(f);
  if (n < 0) {
    fclose(f);
    return NULL;
  }
  rewind(f);
  char *buf = malloc((size_t)n + 1);
  if (!buf) {
    fclose(f);
    return NULL;
  }
  size_t got = fread(buf, 1, (size_t)n, f);
  fclose(f);
  if (got != (size_t)n) {
    free(buf);
    return NULL;
  }
  buf[got] = '\0';
  if (len_out) *len_out = got;
  return buf;
}
