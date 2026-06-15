#define _POSIX_C_SOURCE 200809L
#include "stt/audio.h"

#include <assert.h>
#include <stdlib.h>

static void test_long_capture_segments_at_max_audio(void) {
  SttConfig config = {
    .max_audio_sec = 1,
    .pre_roll_ms = 0,
    .post_roll_ms = 0,
  };
  SttRecorder *rec = NULL;
  assert(stt_recorder_test_new(&rec, &config) == 0);
  assert(stt_recorder_begin(rec) == 0);

  size_t sample_count = (size_t)STT_SAMPLE_RATE * 2 + (size_t)STT_SAMPLE_RATE / 4;
  int16_t *samples = calloc(sample_count, sizeof(*samples));
  assert(samples);
  for (size_t i = 0; i < sample_count; ++i) samples[i] = (int16_t)(i % 3000);

  assert(stt_recorder_test_push(rec, samples, sample_count) == 0);

  SttAudioSegment *segments = NULL;
  assert(stt_recorder_commit_segments(rec, &segments) == 0);

  size_t lengths[] = {(size_t)STT_SAMPLE_RATE, (size_t)STT_SAMPLE_RATE, (size_t)STT_SAMPLE_RATE / 4};
  SttAudioSegmentReason reasons[] = {
    STT_AUDIO_SEGMENT_MAX_AUDIO,
    STT_AUDIO_SEGMENT_MAX_AUDIO,
    STT_AUDIO_SEGMENT_RELEASE,
  };
  size_t total = 0;
  unsigned int count = 0;
  for (SttAudioSegment *segment = segments; segment; segment = segment->next) {
    assert(count < 3);
    assert(segment->index == count);
    assert(segment->audio.len == lengths[count]);
    assert(segment->reason == reasons[count]);
    total += segment->audio.len;
    count++;
  }
  assert(count == 3);
  assert(total == sample_count);

  stt_audio_segments_free(segments);
  stt_recorder_test_free(rec);
  free(samples);
}

int main(void) {
  test_long_capture_segments_at_max_audio();
  return 0;
}
