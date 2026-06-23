#define _POSIX_C_SOURCE 200809L
#include "stt/audio.h"
#include "stt/log.h"

#include "backend.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct SttRecorder {
  SttAudioSource *source;
  pthread_t thread;
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int stop_thread;
  int active;
  int committing;
  int read_error;
  int sample_rate;
  int max_samples;
  int pre_roll_samples;
  int post_roll_samples;
  size_t post_remaining_samples;
  int16_t *pre;
  size_t pre_cap;
  size_t pre_len;
  size_t pre_pos;
  SttAudioBuffer capture;
  SttAudioSegment *pending_head;
  SttAudioSegment *pending_tail;
  unsigned int next_segment_index;
  SttAudioMonitor monitor;
  void *monitor_user;
  int owns_thread;
};

void stt_audio_buffer_init(SttAudioBuffer *buf) {
  memset(buf, 0, sizeof(*buf));
  buf->sample_rate = 16000;
  buf->channels = 1;
}

void stt_audio_buffer_free(SttAudioBuffer *buf) {
  free(buf->samples);
  memset(buf, 0, sizeof(*buf));
}

void stt_audio_segments_free(SttAudioSegment *segment) {
  while (segment) {
    SttAudioSegment *next = segment->next;
    stt_audio_buffer_free(&segment->audio);
    free(segment);
    segment = next;
  }
}

static int append_samples(SttAudioBuffer *buf, const int16_t *samples, size_t n, size_t max_len) {
  if (max_len && buf->len + n > max_len) n = max_len - buf->len;
  if (n == 0) return 0;
  if (buf->len + n > buf->cap) {
    size_t next_cap = buf->cap ? buf->cap * 2 : 16384;
    while (buf->len + n > next_cap) next_cap *= 2;
    int16_t *next = realloc(buf->samples, next_cap * sizeof(*next));
    if (!next) return -1;
    buf->samples = next;
    buf->cap = next_cap;
  }
  memcpy(buf->samples + buf->len, samples, n * sizeof(*samples));
  buf->len += n;
  return 0;
}

static int append_pending_segment(SttRecorder *rec, SttAudioSegmentReason reason) {
  if (rec->capture.len == 0) return 0;
  SttAudioSegment *segment = calloc(1, sizeof(*segment));
  if (!segment) return -1;
  segment->audio = rec->capture;
  segment->index = rec->next_segment_index++;
  segment->reason = reason;
  stt_audio_buffer_init(&rec->capture);
  rec->capture.sample_rate = rec->sample_rate;
  rec->capture.channels = 1;
  if (rec->pending_tail) rec->pending_tail->next = segment;
  else rec->pending_head = segment;
  rec->pending_tail = segment;
  return 0;
}

static int append_recording_samples(SttRecorder *rec, const int16_t *samples, size_t n) {
  if (rec->max_samples <= 0) return -1;
  size_t offset = 0;
  while (offset < n) {
    size_t room = (size_t)rec->max_samples > rec->capture.len ? (size_t)rec->max_samples - rec->capture.len : 0;
    if (room == 0) {
      if (append_pending_segment(rec, STT_AUDIO_SEGMENT_MAX_AUDIO) != 0) return -1;
      continue;
    }
    size_t take = n - offset;
    if (take > room) take = room;
    if (append_samples(&rec->capture, samples + offset, take, (size_t)rec->max_samples) != 0) return -1;
    offset += take;
    if (rec->capture.len >= (size_t)rec->max_samples) {
      if (append_pending_segment(rec, STT_AUDIO_SEGMENT_MAX_AUDIO) != 0) return -1;
    }
  }
  return 0;
}

static void remember_pre_roll(SttRecorder *rec, const int16_t *samples, size_t n) {
  if (!rec->pre || rec->pre_cap == 0) return;
  if (n >= rec->pre_cap) {
    memcpy(rec->pre, samples + (n - rec->pre_cap), rec->pre_cap * sizeof(*samples));
    rec->pre_pos = 0;
    rec->pre_len = rec->pre_cap;
    return;
  }
  size_t offset = 0;
  while (offset < n) {
    size_t chunk = rec->pre_cap - rec->pre_pos;
    if (chunk > n - offset) chunk = n - offset;
    memcpy(rec->pre + rec->pre_pos, samples + offset, chunk * sizeof(*samples));
    rec->pre_pos = (rec->pre_pos + chunk) % rec->pre_cap;
    rec->pre_len += chunk;
    if (rec->pre_len > rec->pre_cap) rec->pre_len = rec->pre_cap;
    offset += chunk;
  }
}

static int copy_pre_roll(SttRecorder *rec) {
  if (!rec->pre || rec->pre_len == 0) return 0;
  size_t start = rec->pre_len == rec->pre_cap ? rec->pre_pos : 0;
  size_t first = rec->pre_cap - start;
  if (first > rec->pre_len) first = rec->pre_len;
  if (append_samples(&rec->capture, rec->pre + start, first, (size_t)rec->max_samples) != 0) return -1;
  size_t remaining = rec->pre_len - first;
  if (remaining > 0 && append_samples(&rec->capture, rec->pre, remaining, (size_t)rec->max_samples) != 0) return -1;
  return 0;
}

static void *record_thread(void *arg) {
  SttRecorder *rec = arg;
  int16_t chunk[256];
  while (!rec->stop_thread) {
    if (stt_audio_source_read(rec->source, chunk, sizeof(chunk) / sizeof(chunk[0])) != 0) {
      pthread_mutex_lock(&rec->mutex);
      rec->read_error = 1;
      rec->active = 0;
      rec->committing = 0;
      pthread_cond_broadcast(&rec->cond);
      pthread_mutex_unlock(&rec->mutex);
      LOG_ERROR("%s read failed\n", stt_audio_backend_name());
      break;
    }

    size_t n = sizeof(chunk) / sizeof(chunk[0]);
    pthread_mutex_lock(&rec->mutex);
    if (rec->monitor) rec->monitor(chunk, n, rec->monitor_user);
    remember_pre_roll(rec, chunk, n);
    if (rec->active) {
      if (append_recording_samples(rec, chunk, n) != 0) {
        rec->read_error = 1;
        rec->active = 0;
        rec->committing = 0;
        pthread_cond_broadcast(&rec->cond);
      } else if (rec->committing) {
        if (rec->post_remaining_samples > n) {
          rec->post_remaining_samples -= n;
        } else {
          rec->post_remaining_samples = 0;
          rec->active = 0;
          rec->committing = 0;
          pthread_cond_broadcast(&rec->cond);
        }
      }
    }
    pthread_mutex_unlock(&rec->mutex);
  }
  return NULL;
}

int stt_recorder_open(SttRecorder **out, const SttConfig *config) {
  SttRecorder *rec = calloc(1, sizeof(*rec));
  if (!rec) return -1;
  rec->sample_rate = STT_SAMPLE_RATE;
  rec->max_samples = STT_SAMPLE_RATE * config->max_audio_sec;
  rec->pre_roll_samples = STT_SAMPLE_RATE * config->pre_roll_ms / 1000;
  rec->post_roll_samples = STT_SAMPLE_RATE * config->post_roll_ms / 1000;
  stt_audio_buffer_init(&rec->capture);
  rec->capture.sample_rate = STT_SAMPLE_RATE;
  rec->pre_cap = (size_t)rec->pre_roll_samples;
  if (rec->pre_cap > 0) {
    rec->pre = calloc(rec->pre_cap, sizeof(*rec->pre));
    if (!rec->pre) {
      free(rec);
      return -1;
    }
  }
  pthread_mutex_init(&rec->mutex, NULL);
  pthread_cond_init(&rec->cond, NULL);

  if (stt_audio_source_open(&rec->source, config, "dictation") != 0) {
    free(rec->pre);
    free(rec);
    return -1;
  }
  if (pthread_create(&rec->thread, NULL, record_thread, rec) != 0) {
    stt_audio_source_close(rec->source);
    free(rec->pre);
    free(rec);
    return -1;
  }
  rec->owns_thread = 1;
  LOG_INFO("Microphone ready: backend=%s %d Hz mono, %.2fs pre-roll, %.2fs post-roll, %.0fs segments\n",
          stt_audio_backend_name(),
          STT_SAMPLE_RATE,
          rec->pre_roll_samples / (double)STT_SAMPLE_RATE,
          rec->post_roll_samples / (double)STT_SAMPLE_RATE,
          rec->max_samples / (double)STT_SAMPLE_RATE);
  LOG_DEBUG("audio monitor ready: fragment %.3fs\n", 256.0 / (double)STT_SAMPLE_RATE);
  *out = rec;
  return 0;
}

void stt_recorder_set_monitor(SttRecorder *rec, SttAudioMonitor monitor, void *user) {
  if (!rec) return;
  pthread_mutex_lock(&rec->mutex);
  rec->monitor = monitor;
  rec->monitor_user = user;
  pthread_mutex_unlock(&rec->mutex);
}

void stt_recorder_close(SttRecorder *rec) {
  if (!rec) return;
  if (rec->owns_thread) {
    rec->stop_thread = 1;
    pthread_join(rec->thread, NULL);
  }
  stt_audio_source_close(rec->source);
  pthread_mutex_destroy(&rec->mutex);
  pthread_cond_destroy(&rec->cond);
  stt_audio_buffer_free(&rec->capture);
  stt_audio_segments_free(rec->pending_head);
  free(rec->pre);
  free(rec);
}

int stt_recorder_begin(SttRecorder *rec) {
  if (!rec) return -1;
  pthread_mutex_lock(&rec->mutex);
  rec->capture.len = 0;
  rec->capture.sample_rate = rec->sample_rate;
  rec->capture.channels = 1;
  stt_audio_segments_free(rec->pending_head);
  rec->pending_head = NULL;
  rec->pending_tail = NULL;
  rec->next_segment_index = 0;
  rec->active = 1;
  rec->committing = 0;
  rec->post_remaining_samples = 0;
  int rc = copy_pre_roll(rec);
  double pre = rec->capture.len / (double)rec->sample_rate;
  pthread_mutex_unlock(&rec->mutex);
  LOG_DEBUG("recording... included %.2fs pre-roll (%zu samples)\n", pre, rec->capture.len);
  return rc;
}

int stt_recorder_commit_segments(SttRecorder *rec, SttAudioSegment **out) {
  if (!rec || !out) return -1;
  *out = NULL;
  pthread_mutex_lock(&rec->mutex);
  if (rec->active) {
    if (rec->post_roll_samples <= 0) {
      rec->active = 0;
      rec->committing = 0;
      LOG_DEBUG("release detected; stopping immediately with no post-roll\n");
    } else {
      rec->committing = 1;
      rec->post_remaining_samples = (size_t)rec->post_roll_samples;
      LOG_DEBUG("release detected; collecting %.2fs post-roll current_samples=%zu\n",
              rec->post_roll_samples / (double)rec->sample_rate, rec->capture.len);
      while (rec->active && rec->committing && !rec->read_error) {
        pthread_cond_wait(&rec->cond, &rec->mutex);
      }
    }
  }
  if (rec->capture.len > 0) {
    if (append_pending_segment(rec, STT_AUDIO_SEGMENT_RELEASE) != 0) {
      pthread_mutex_unlock(&rec->mutex);
      return -1;
    }
  }
  SttAudioSegment *segments = rec->pending_head;
  rec->pending_head = NULL;
  rec->pending_tail = NULL;
  size_t samples = 0;
  for (SttAudioSegment *segment = segments; segment; segment = segment->next) samples += segment->audio.len;
  *out = segments;
  double seconds = samples / (double)rec->sample_rate;
  pthread_mutex_unlock(&rec->mutex);
  LOG_DEBUG("committed %.2fs audio (%zu samples)\n", seconds, samples);
  return rec->read_error ? -1 : 0;
}

#ifdef STT_TESTING
int stt_recorder_test_new(SttRecorder **out, const SttConfig *config) {
  if (!out || !config) return -1;
  SttRecorder *rec = calloc(1, sizeof(*rec));
  if (!rec) return -1;
  rec->sample_rate = STT_SAMPLE_RATE;
  rec->max_samples = STT_SAMPLE_RATE * config->max_audio_sec;
  rec->pre_roll_samples = STT_SAMPLE_RATE * config->pre_roll_ms / 1000;
  rec->post_roll_samples = STT_SAMPLE_RATE * config->post_roll_ms / 1000;
  stt_audio_buffer_init(&rec->capture);
  rec->capture.sample_rate = STT_SAMPLE_RATE;
  pthread_mutex_init(&rec->mutex, NULL);
  pthread_cond_init(&rec->cond, NULL);
  *out = rec;
  return 0;
}

void stt_recorder_test_free(SttRecorder *rec) {
  stt_recorder_close(rec);
}

int stt_recorder_test_push(SttRecorder *rec, const int16_t *samples, size_t n) {
  if (!rec) return -1;
  pthread_mutex_lock(&rec->mutex);
  int rc = append_recording_samples(rec, samples, n);
  pthread_mutex_unlock(&rec->mutex);
  return rc;
}
#endif
