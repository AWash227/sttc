#define _POSIX_C_SOURCE 200809L

#include "backend.h"

#include <stdio.h>

const char *stt_text_backend_name(void) {
  return "stdout";
}

int stt_text_output_write(const char *text, int delay_ms) {
  (void)delay_ms;
  if (!text) return -1;
  printf("%s\n", text);
  fflush(stdout);
  return 0;
}
