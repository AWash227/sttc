#define _POSIX_C_SOURCE 200809L
#include "stt/log.h"

#include <stdarg.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int g_log_initialized;
static SttLogLevel g_log_level = STT_LOG_INFO;

long long stt_now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static SttLogLevel parse_level(const char *value) {
  if (!value || !*value) return STT_LOG_INFO;
  char lower[16];
  size_t i = 0;
  for (; value[i] && i + 1 < sizeof(lower); ++i) {
    lower[i] = (char)tolower((unsigned char)value[i]);
  }
  lower[i] = '\0';
  if (strcmp(lower, "error") == 0) return STT_LOG_ERROR;
  if (strcmp(lower, "warn") == 0 || strcmp(lower, "warning") == 0) return STT_LOG_WARN;
  if (strcmp(lower, "info") == 0) return STT_LOG_INFO;
  if (strcmp(lower, "debug") == 0) return STT_LOG_DEBUG;
  if (strcmp(lower, "trace") == 0) return STT_LOG_TRACE;
  return STT_LOG_INFO;
}

static void init_log(void) {
  if (g_log_initialized) return;
  const char *value = getenv("STT_LOG_LEVEL");
  if (!value || !*value) value = getenv("STT_LOG");
  g_log_level = parse_level(value);
  g_log_initialized = 1;
}

int stt_log_enabled(SttLogLevel level) {
  init_log();
  return level <= g_log_level;
}

void stt_log(SttLogLevel level, const char *fmt, ...) {
  if (!stt_log_enabled(level)) return;
  va_list ap;
  va_start(ap, fmt);
  vfprintf(stderr, fmt, ap);
  va_end(ap);
}
