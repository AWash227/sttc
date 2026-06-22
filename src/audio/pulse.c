#define _POSIX_C_SOURCE 200809L
#include "stt/log.h"

#include "backend.h"

#include <pulse/error.h>
#include <pulse/simple.h>
#include <stdlib.h>

struct SttAudioSource {
  pa_simple *pa;
};

const char *stt_audio_backend_name(void) {
  return "pulse";
}

int stt_audio_source_open(SttAudioSource **out, const SttConfig *config, const char *stream_name) {
  (void)config;
  if (!out) return -1;
  *out = NULL;
  SttAudioSource *source = calloc(1, sizeof(*source));
  if (!source) return -1;
  pa_sample_spec ss = {
    .format = PA_SAMPLE_S16LE,
    .rate = STT_SAMPLE_RATE,
    .channels = 1,
  };
  pa_buffer_attr attr = {
    .maxlength = (uint32_t)-1,
    .tlength = (uint32_t)-1,
    .prebuf = (uint32_t)-1,
    .minreq = (uint32_t)-1,
    .fragsize = 256 * sizeof(int16_t),
  };
  int error = 0;
  source->pa = pa_simple_new(NULL, "stt", PA_STREAM_RECORD, NULL, stream_name, &ss, NULL, &attr, &error);
  if (!source->pa) {
    LOG_ERROR("PulseAudio open failed: %s\n", pa_strerror(error));
    free(source);
    return -1;
  }
  *out = source;
  return 0;
}

void stt_audio_source_close(SttAudioSource *source) {
  if (!source) return;
  if (source->pa) pa_simple_free(source->pa);
  free(source);
}

int stt_audio_source_read(SttAudioSource *source, int16_t *samples, size_t sample_count) {
  int error = 0;
  if (!source || !source->pa || !samples) return -1;
  if (pa_simple_read(source->pa, samples, sample_count * sizeof(*samples), &error) < 0) {
    LOG_ERROR("PulseAudio read failed: %s\n", pa_strerror(error));
    return -1;
  }
  return 0;
}
