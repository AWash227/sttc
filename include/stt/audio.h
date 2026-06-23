#ifndef STT_AUDIO_H
#define STT_AUDIO_H

#include "stt/config.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
  int16_t *samples;
  size_t len;
  size_t cap;
  int sample_rate;
  int channels;
} SttAudioBuffer;

typedef enum {
  STT_AUDIO_SEGMENT_RELEASE = 0,
  STT_AUDIO_SEGMENT_MAX_AUDIO = 1,
} SttAudioSegmentReason;

typedef struct SttAudioSegment {
  SttAudioBuffer audio;
  unsigned int index;
  SttAudioSegmentReason reason;
  struct SttAudioSegment *next;
} SttAudioSegment;

typedef struct SttRecorder SttRecorder;
typedef void (*SttAudioMonitor)(const int16_t *samples, size_t sample_count, void *user);

void stt_audio_buffer_init(SttAudioBuffer *buf);
void stt_audio_buffer_free(SttAudioBuffer *buf);
void stt_audio_segments_free(SttAudioSegment *segment);

int stt_recorder_open(SttRecorder **out, const SttConfig *config);
void stt_recorder_close(SttRecorder *rec);
void stt_recorder_set_monitor(SttRecorder *rec, SttAudioMonitor monitor, void *user);
int stt_recorder_begin(SttRecorder *rec);
int stt_recorder_commit_segments(SttRecorder *rec, SttAudioSegment **out);

#ifdef STT_TESTING
int stt_recorder_test_new(SttRecorder **out, const SttConfig *config);
void stt_recorder_test_free(SttRecorder *rec);
int stt_recorder_test_push(SttRecorder *rec, const int16_t *samples, size_t n);
#endif

#endif
