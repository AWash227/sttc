#define _POSIX_C_SOURCE 200809L
#include "stt/audio.h"

#include <pulse/error.h>
#include <pulse/simple.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct SttRecorder {
  pa_simple *pa;
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
  size_t post_target_len;
  int16_t *pre;
  size_t pre_cap;
  size_t pre_len;
  size_t pre_pos;
  SttAudioBuffer capture;
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

static void remember_pre_roll(SttRecorder *rec, const int16_t *samples, size_t n) {
  if (!rec->pre || rec->pre_cap == 0) return;
  for (size_t i = 0; i < n; ++i) {
    rec->pre[rec->pre_pos] = samples[i];
    rec->pre_pos = (rec->pre_pos + 1) % rec->pre_cap;
    if (rec->pre_len < rec->pre_cap) rec->pre_len++;
  }
}

static int copy_pre_roll(SttRecorder *rec) {
  if (!rec->pre || rec->pre_len == 0) return 0;
  size_t start = rec->pre_len == rec->pre_cap ? rec->pre_pos : 0;
  for (size_t i = 0; i < rec->pre_len; ++i) {
    int16_t sample = rec->pre[(start + i) % rec->pre_cap];
    if (append_samples(&rec->capture, &sample, 1, (size_t)rec->max_samples) != 0) return -1;
  }
  return 0;
}

static void *record_thread(void *arg) {
  SttRecorder *rec = arg;
  int16_t chunk[256];
  while (!rec->stop_thread) {
    int error = 0;
    if (pa_simple_read(rec->pa, chunk, sizeof(chunk), &error) < 0) {
      pthread_mutex_lock(&rec->mutex);
      rec->read_error = 1;
      rec->active = 0;
      rec->committing = 0;
      pthread_cond_broadcast(&rec->cond);
      pthread_mutex_unlock(&rec->mutex);
      fprintf(stderr, "PulseAudio read failed: %s\n", pa_strerror(error));
      break;
    }

    size_t n = sizeof(chunk) / sizeof(chunk[0]);
    pthread_mutex_lock(&rec->mutex);
    remember_pre_roll(rec, chunk, n);
    if (rec->active) {
      append_samples(&rec->capture, chunk, n, (size_t)rec->max_samples);
      if ((int)rec->capture.len >= rec->max_samples) {
        rec->active = 0;
        rec->committing = 0;
        pthread_cond_broadcast(&rec->cond);
      } else if (rec->committing && rec->capture.len >= rec->post_target_len) {
          rec->active = 0;
          rec->committing = 0;
          pthread_cond_broadcast(&rec->cond);
      }
    }
    pthread_mutex_unlock(&rec->mutex);
  }
  return NULL;
}

int stt_recorder_open(SttRecorder **out, int sample_rate, int max_seconds, int pre_roll_ms, int post_roll_ms) {
  SttRecorder *rec = calloc(1, sizeof(*rec));
  if (!rec) return -1;
  rec->sample_rate = sample_rate;
  rec->max_samples = sample_rate * max_seconds;
  rec->pre_roll_samples = sample_rate * pre_roll_ms / 1000;
  rec->post_roll_samples = sample_rate * post_roll_ms / 1000;
  stt_audio_buffer_init(&rec->capture);
  rec->capture.sample_rate = sample_rate;
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

  pa_sample_spec ss = {
    .format = PA_SAMPLE_S16LE,
    .rate = (uint32_t)sample_rate,
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
  rec->pa = pa_simple_new(NULL, "stt", PA_STREAM_RECORD, NULL, "dictation", &ss, NULL, &attr, &error);
  if (!rec->pa) {
    fprintf(stderr, "PulseAudio open failed: %s\n", pa_strerror(error));
    free(rec->pre);
    free(rec);
    return -1;
  }
  if (pthread_create(&rec->thread, NULL, record_thread, rec) != 0) {
    pa_simple_free(rec->pa);
    free(rec->pre);
    free(rec);
    return -1;
  }
  fprintf(stderr, "audio monitor ready: %d Hz mono, pre-roll %.2fs, post-roll %.2fs, max %.2fs, fragment %.3fs\n",
          sample_rate,
          rec->pre_roll_samples / (double)sample_rate,
          rec->post_roll_samples / (double)sample_rate,
          rec->max_samples / (double)sample_rate,
          attr.fragsize / (double)(sample_rate * sizeof(int16_t)));
  *out = rec;
  return 0;
}

void stt_recorder_close(SttRecorder *rec) {
  if (!rec) return;
  rec->stop_thread = 1;
  pthread_join(rec->thread, NULL);
  if (rec->pa) pa_simple_free(rec->pa);
  pthread_mutex_destroy(&rec->mutex);
  pthread_cond_destroy(&rec->cond);
  stt_audio_buffer_free(&rec->capture);
  free(rec->pre);
  free(rec);
}

int stt_recorder_begin(SttRecorder *rec) {
  if (!rec) return -1;
  pthread_mutex_lock(&rec->mutex);
  rec->capture.len = 0;
  rec->capture.sample_rate = rec->sample_rate;
  rec->capture.channels = 1;
  rec->active = 1;
  rec->committing = 0;
  rec->post_target_len = 0;
  int rc = copy_pre_roll(rec);
  double pre = rec->capture.len / (double)rec->sample_rate;
  pthread_mutex_unlock(&rec->mutex);
  fprintf(stderr, "recording... included %.2fs pre-roll (%zu samples)\n", pre, rec->capture.len);
  return rc;
}

int stt_recorder_commit(SttRecorder *rec, SttAudioBuffer *out) {
  if (!rec || !out) return -1;
  stt_audio_buffer_init(out);
  pthread_mutex_lock(&rec->mutex);
  if (rec->active) {
    if (rec->post_roll_samples <= 0) {
      rec->active = 0;
      rec->committing = 0;
      fprintf(stderr, "release detected; stopping immediately with no post-roll\n");
    } else {
      rec->committing = 1;
      rec->post_target_len = rec->capture.len + (size_t)rec->post_roll_samples;
      fprintf(stderr, "release detected; collecting %.2fs post-roll target_samples=%zu current_samples=%zu\n",
              rec->post_roll_samples / (double)rec->sample_rate, rec->post_target_len, rec->capture.len);
      while (rec->active && rec->committing && !rec->read_error) {
        pthread_cond_wait(&rec->cond, &rec->mutex);
      }
    }
  }
  out->sample_rate = rec->sample_rate;
  out->channels = 1;
  if (rec->capture.len > 0) {
    out->samples = malloc(rec->capture.len * sizeof(*out->samples));
    if (!out->samples) {
      pthread_mutex_unlock(&rec->mutex);
      return -1;
    }
    memcpy(out->samples, rec->capture.samples, rec->capture.len * sizeof(*out->samples));
    out->len = rec->capture.len;
    out->cap = rec->capture.len;
  }
  double seconds = out->len / (double)out->sample_rate;
  pthread_mutex_unlock(&rec->mutex);
  fprintf(stderr, "committed %.2fs audio (%zu samples)\n", seconds, out->len);
  return rec->read_error ? -1 : 0;
}
