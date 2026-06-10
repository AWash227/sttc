#ifndef STT_AUDIO_H
#define STT_AUDIO_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
  int16_t *samples;
  size_t len;
  size_t cap;
  int sample_rate;
  int channels;
} SttAudioBuffer;

typedef struct SttRecorder SttRecorder;

void stt_audio_buffer_init(SttAudioBuffer *buf);
void stt_audio_buffer_free(SttAudioBuffer *buf);

int stt_recorder_open(SttRecorder **out, int sample_rate, int max_seconds, int pre_roll_ms, int post_roll_ms);
void stt_recorder_close(SttRecorder *rec);
int stt_recorder_begin(SttRecorder *rec);
int stt_recorder_commit(SttRecorder *rec, SttAudioBuffer *out);

#endif
