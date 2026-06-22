#include "stt/log.h"

#include "backend.h"

const char *stt_audio_backend_name(void) {
  return "none";
}

int stt_audio_source_open(SttAudioSource **out, const SttConfig *config, const char *stream_name) {
  (void)out;
  (void)config;
  (void)stream_name;
  LOG_ERROR("audio backend is disabled in this build\n");
  return -1;
}

void stt_audio_source_close(SttAudioSource *source) {
  (void)source;
}

int stt_audio_source_read(SttAudioSource *source, int16_t *samples, size_t sample_count) {
  (void)source;
  (void)samples;
  (void)sample_count;
  return -1;
}
