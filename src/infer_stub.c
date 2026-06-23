#include "stt/infer.h"
#include "stt/log.h"

#include <stdlib.h>
#include <string.h>

static char *empty_text(void) {
  char *s = malloc(1);
  if (s) s[0] = '\0';
  return s;
}

int stt_transcribe_warmup(SttModel *model, const SttConfig *config) {
  (void)model;
  (void)config;
  LOG_WARN("inference is disabled in this build\n");
  return -1;
}

int stt_transcribe(SttModel *model, const SttAudioBuffer *audio, const SttConfig *config, char **text_out) {
  (void)model;
  (void)audio;
  (void)config;
  if (text_out) *text_out = empty_text();
  return -1;
}
