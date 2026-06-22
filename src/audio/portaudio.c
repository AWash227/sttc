#define _POSIX_C_SOURCE 200809L
#include "stt/log.h"

#include "backend.h"

#include <portaudio.h>
#include <stdlib.h>

struct SttAudioSource {
  PaStream *stream;
};

const char *stt_audio_backend_name(void) {
  return "portaudio";
}

int stt_audio_source_open(SttAudioSource **out, const SttConfig *config, const char *stream_name) {
  (void)config;
  (void)stream_name;
  if (!out) return -1;
  *out = NULL;
  PaError err = Pa_Initialize();
  if (err != paNoError) {
    LOG_ERROR("PortAudio initialize failed: %s\n", Pa_GetErrorText(err));
    return -1;
  }
  SttAudioSource *source = calloc(1, sizeof(*source));
  if (!source) {
    Pa_Terminate();
    return -1;
  }
  err = Pa_OpenDefaultStream(&source->stream, 1, 0, paInt16, STT_SAMPLE_RATE, 256, NULL, NULL);
  if (err == paNoError) err = Pa_StartStream(source->stream);
  if (err != paNoError) {
    LOG_ERROR("PortAudio open failed: %s\n", Pa_GetErrorText(err));
    if (source->stream) Pa_CloseStream(source->stream);
    free(source);
    Pa_Terminate();
    return -1;
  }
  *out = source;
  return 0;
}

void stt_audio_source_close(SttAudioSource *source) {
  if (!source) return;
  if (source->stream) {
    Pa_StopStream(source->stream);
    Pa_CloseStream(source->stream);
  }
  Pa_Terminate();
  free(source);
}

int stt_audio_source_read(SttAudioSource *source, int16_t *samples, size_t sample_count) {
  if (!source || !source->stream || !samples) return -1;
  PaError err = Pa_ReadStream(source->stream, samples, (unsigned long)sample_count);
  if (err != paNoError) {
    LOG_ERROR("PortAudio read failed: %s\n", Pa_GetErrorText(err));
    return -1;
  }
  return 0;
}
