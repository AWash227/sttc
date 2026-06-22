#define _POSIX_C_SOURCE 200809L
#include "stt/audio.h"
#include "stt/cli.h"
#include "stt/infer.h"
#include "stt/log.h"
#include "stt/model.h"

#include "audio/backend.h"
#include "hotkey/backend.h"
#include "text/backend.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef enum {
  TRANSCRIBE_JOB_HOTKEY,
  TRANSCRIBE_JOB_LOG,
} TranscribeJobKind;

typedef struct TranscribeJob {
  SttAudioBuffer audio;
  TranscribeJobKind kind;
  unsigned long long capture_id;
  unsigned int segment_index;
  unsigned int segment_count;
  SttAudioSegmentReason segment_reason;
  long long release_ms;
  long long enqueue_ms;
  long long commit_ms;
  struct TranscribeJob *next;
} TranscribeJob;

typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  pthread_t thread;
  TranscribeJob *head;
  TranscribeJob *tail;
  const SttConfig *config;
  SttModel *model;
  atomic_int recording;
  size_t depth;
  int stop;
  int started;
} TranscribeQueue;

typedef struct {
  const SttConfig *config;
  SttRecorder *recorder;
  TranscribeQueue *queue;
  int recording;
  unsigned long long event_seq;
  unsigned long long capture_seq;
  long long capture_begin_ms;
} RunState;

typedef struct {
  const SttConfig *config;
  TranscribeQueue *queue;
  pthread_t thread;
  atomic_int stop;
  int started;
  unsigned long long capture_seq;
} AutoLogState;

typedef struct {
  int16_t *samples;
  size_t cap;
  size_t len;
  size_t pos;
} SampleRing;

typedef struct {
  int peak;
  int avg;
} AudioLevel;

typedef struct {
  double floor;
  int initialized;
} VadState;

static double samples_to_seconds(size_t samples) {
  return samples / (double)STT_SAMPLE_RATE;
}

static size_t transcribe_queue_depth(TranscribeQueue *queue) {
  pthread_mutex_lock(&queue->mutex);
  size_t depth = queue->depth;
  pthread_mutex_unlock(&queue->mutex);
  return depth;
}

static int transcribe_queue_push(TranscribeQueue *queue,
                                 TranscribeJobKind kind,
                                 unsigned long long capture_id,
                                 unsigned int segment_index,
                                 unsigned int segment_count,
                                 SttAudioSegmentReason segment_reason,
                                 long long release_ms,
                                 long long commit_ms,
                                 SttAudioBuffer *audio,
                                 size_t *depth_out);

static void append_transcript_line(const char *path, const char *text) {
  FILE *f = fopen(path, "a");
  if (!f) {
    LOG_ERROR("log: failed to open %s\n", path);
    return;
  }
  fprintf(f, "%s\n", text);
  fclose(f);
}

static void ring_free(SampleRing *ring) {
  free(ring->samples);
  memset(ring, 0, sizeof(*ring));
}

static int ring_init(SampleRing *ring, size_t cap) {
  memset(ring, 0, sizeof(*ring));
  ring->samples = calloc(cap, sizeof(*ring->samples));
  if (!ring->samples) return -1;
  ring->cap = cap;
  return 0;
}

static void ring_push(SampleRing *ring, const int16_t *samples, size_t n) {
  if (ring->cap == 0) return;
  if (n >= ring->cap) {
    memcpy(ring->samples, samples + n - ring->cap, ring->cap * sizeof(*samples));
    ring->pos = 0;
    ring->len = ring->cap;
    return;
  }
  size_t offset = 0;
  while (offset < n) {
    size_t chunk = ring->cap - ring->pos;
    if (chunk > n - offset) chunk = n - offset;
    memcpy(ring->samples + ring->pos, samples + offset, chunk * sizeof(*samples));
    ring->pos = (ring->pos + chunk) % ring->cap;
    ring->len += chunk;
    if (ring->len > ring->cap) ring->len = ring->cap;
    offset += chunk;
  }
}

static int append_audio(SttAudioBuffer *audio, const int16_t *samples, size_t n, size_t max_len) {
  if (max_len && audio->len + n > max_len) n = max_len - audio->len;
  if (n == 0) return 0;
  if (audio->len + n > audio->cap) {
    size_t next_cap = audio->cap ? audio->cap * 2 : 16384;
    while (audio->len + n > next_cap) next_cap *= 2;
    int16_t *next = realloc(audio->samples, next_cap * sizeof(*next));
    if (!next) return -1;
    audio->samples = next;
    audio->cap = next_cap;
  }
  memcpy(audio->samples + audio->len, samples, n * sizeof(*samples));
  audio->len += n;
  return 0;
}

static int append_ring_audio(SttAudioBuffer *audio, const SampleRing *ring, size_t max_len) {
  if (ring->len == 0) return 0;
  size_t start = ring->len == ring->cap ? ring->pos : 0;
  size_t first = ring->cap - start;
  if (first > ring->len) first = ring->len;
  if (append_audio(audio, ring->samples + start, first, max_len) != 0) return -1;
  size_t remaining = ring->len - first;
  if (remaining > 0 && append_audio(audio, ring->samples, remaining, max_len) != 0) return -1;
  return 0;
}

static int int_max(int a, int b) {
  return a > b ? a : b;
}

static int vad_threshold(const VadState *vad, double multiplier, int minimum) {
  double floor = vad->initialized ? vad->floor : 50.0;
  int threshold = (int)(floor * multiplier);
  return int_max(threshold, minimum);
}

static AudioLevel chunk_level(const int16_t *samples, size_t n) {
  int peak = 0;
  long long sum = 0;
  for (size_t i = 0; i < n; ++i) {
    int v = samples[i] < 0 ? -samples[i] : samples[i];
    if (v > peak) peak = v;
    sum += v;
  }
  AudioLevel level = {.peak = peak, .avg = n > 0 ? (int)(sum / (long long)n) : 0};
  return level;
}

static void vad_update_floor(VadState *vad, AudioLevel level) {
  if (!vad->initialized) {
    vad->floor = level.avg;
    vad->initialized = 1;
    return;
  }
  double alpha = level.avg < vad->floor ? 0.15 : 0.02;
  vad->floor = vad->floor * (1.0 - alpha) + (double)level.avg * alpha;
  if (vad->floor < 5.0) vad->floor = 5.0;
}

static int vad_is_speech(const VadState *vad, AudioLevel level) {
  int avg_threshold = vad_threshold(vad, 3.5, 140);
  int peak_threshold = vad_threshold(vad, 12.0, 900);
  return level.avg >= avg_threshold || level.peak >= peak_threshold;
}

static int vad_is_silence(const VadState *vad, AudioLevel level) {
  int avg_threshold = vad_threshold(vad, 2.4, 110);
  int peak_threshold = vad_threshold(vad, 8.0, 650);
  return level.avg < avg_threshold && level.peak < peak_threshold;
}

static void enqueue_log_audio(AutoLogState *state, SttAudioBuffer *audio) {
  size_t min_samples = (size_t)(STT_SAMPLE_RATE * 350 / 1000);
  if (audio->len <= min_samples) {
    stt_audio_buffer_free(audio);
    stt_audio_buffer_init(audio);
    return;
  }
  unsigned long long capture_id = ++state->capture_seq;
  long long now = stt_now_ms();
  size_t depth = 0;
  size_t samples = audio->len;
  if (transcribe_queue_push(state->queue, TRANSCRIBE_JOB_LOG, capture_id, 0, 1, STT_AUDIO_SEGMENT_RELEASE, now, 0, audio, &depth) != 0) {
    LOG_ERROR("log: capture=%llu discarded reason=queue_alloc_failed\n", capture_id);
    stt_audio_buffer_free(audio);
    stt_audio_buffer_init(audio);
    return;
  }
  LOG_DEBUG("log: capture=%llu queued samples=%zu queue_depth=%zu\n", capture_id, samples, depth);
}

static void *auto_log_worker(void *arg) {
  AutoLogState *state = arg;
  SttAudioSource *source = NULL;
  if (stt_audio_source_open(&source, state->config, "transcription log") != 0) {
    return NULL;
  }

  SampleRing pre;
  if (ring_init(&pre, (size_t)STT_SAMPLE_RATE * 350 / 1000) != 0) {
    stt_audio_source_close(source);
    return NULL;
  }

  const int silence_chunks_to_stop = 45;
  const size_t max_samples = (size_t)STT_SAMPLE_RATE * (size_t)state->config->max_audio_sec;
  int16_t chunk[256];
  SttAudioBuffer audio;
  stt_audio_buffer_init(&audio);
  VadState vad = {0};
  int active = 0;
  int silence_chunks = 0;
  int trace_chunks = 0;

  LOG_INFO("Continuous log enabled: %s\n", state->config->log_path);
  while (!atomic_load(&state->stop)) {
    if (stt_audio_source_read(source, chunk, sizeof(chunk) / sizeof(chunk[0])) != 0) {
      LOG_ERROR("log: %s read failed\n", stt_audio_backend_name());
      break;
    }

    size_t n = sizeof(chunk) / sizeof(chunk[0]);
    AudioLevel level = chunk_level(chunk, n);
    int speech = vad_is_speech(&vad, level);
    if (!active) {
      ring_push(&pre, chunk, n);
      if (!speech) {
        vad_update_floor(&vad, level);
        continue;
      }
      active = 1;
      silence_chunks = 0;
      stt_audio_buffer_free(&audio);
      stt_audio_buffer_init(&audio);
      append_ring_audio(&audio, &pre, max_samples);
      LOG_DEBUG("log: speech_start avg=%d peak=%d floor=%.1f start_avg=%d start_peak=%d stop_avg=%d stop_peak=%d\n",
                level.avg, level.peak, vad.floor,
                vad_threshold(&vad, 3.5, 140), vad_threshold(&vad, 12.0, 900),
                vad_threshold(&vad, 2.4, 110), vad_threshold(&vad, 8.0, 650));
      continue;
    }

    append_audio(&audio, chunk, n, max_samples);
    int silence = vad_is_silence(&vad, level);
    if (silence) {
      silence_chunks++;
      vad_update_floor(&vad, level);
    } else {
      silence_chunks = 0;
    }

    if (stt_log_enabled(STT_LOG_TRACE) && ++trace_chunks % 64 == 0) {
      LOG_TRACE("log: vad active=%d avg=%d peak=%d floor=%.1f speech=%d silence=%d silence_chunks=%d\n",
                active, level.avg, level.peak, vad.floor, speech, silence, silence_chunks);
    }
    if (silence_chunks >= silence_chunks_to_stop || audio.len >= max_samples) {
      const char *reason = audio.len >= max_samples ? "max_audio" : "silence";
      LOG_DEBUG("log: speech_end reason=%s samples=%zu avg=%d peak=%d floor=%.1f silence_chunks=%d\n",
                reason, audio.len, level.avg, level.peak, vad.floor, silence_chunks);
      enqueue_log_audio(state, &audio);
      active = 0;
      silence_chunks = 0;
    }
  }

  if (active && audio.len > 0) enqueue_log_audio(state, &audio);
  stt_audio_buffer_free(&audio);
  ring_free(&pre);
  stt_audio_source_close(source);
  return NULL;
}

static int auto_log_start(AutoLogState *state, const SttConfig *config, TranscribeQueue *queue) {
  memset(state, 0, sizeof(*state));
  if (!config->log_path) return 0;
  FILE *f = fopen(config->log_path, "a");
  if (!f) {
    LOG_ERROR("log: failed to open %s\n", config->log_path);
    return -1;
  }
  fclose(f);
  state->config = config;
  state->queue = queue;
  atomic_init(&state->stop, 0);
  if (pthread_create(&state->thread, NULL, auto_log_worker, state) != 0) {
    LOG_ERROR("log: failed to start auto transcription thread\n");
    return -1;
  }
  state->started = 1;
  return 0;
}

static void auto_log_stop(AutoLogState *state) {
  if (!state->started) return;
  atomic_store(&state->stop, 1);
  pthread_join(state->thread, NULL);
}

static void *transcribe_worker(void *arg) {
  TranscribeQueue *queue = arg;
  for (;;) {
    pthread_mutex_lock(&queue->mutex);
    while (!queue->head && !queue->stop) pthread_cond_wait(&queue->cond, &queue->mutex);
    if (!queue->head && queue->stop) {
      pthread_mutex_unlock(&queue->mutex);
      return NULL;
    }

    TranscribeJob *job = queue->head;
    queue->head = job->next;
    if (!queue->head) queue->tail = NULL;
    queue->depth--;
    size_t depth_after_pop = queue->depth;
    pthread_mutex_unlock(&queue->mutex);

    LOG_DEBUG("run: capture=%llu segment=%u/%u dequeue wait_ms=%lld queue_depth=%zu\n",
              job->capture_id, job->segment_index + 1, job->segment_count,
              stt_now_ms() - job->enqueue_ms, depth_after_pop);

    char *text = NULL;
    long long transcribe_start_ms = stt_now_ms();
    int rc = stt_transcribe(queue->model, &job->audio, queue->config, &text);
    long long transcribe_ms = stt_now_ms() - transcribe_start_ms;
    long long type_ms = 0;
    LOG_DEBUG("run: capture=%llu segment=%u/%u transcribe rc=%d elapsed_ms=%lld\n",
              job->capture_id, job->segment_index + 1, job->segment_count, rc, transcribe_ms);
    if (text && text[0]) {
      if (job->kind == TRANSCRIBE_JOB_LOG) {
        append_transcript_line(queue->config->log_path, text);
        LOG_INFO("Logged: %s\n", text);
      } else if (queue->config->print_only || queue->config->dry_run) {
        LOG_INFO("Heard: %s\n", text);
        printf("%s\n", text);
        fflush(stdout);
      } else {
        long long type_wait_start_ms = stt_now_ms();
        while (atomic_load(&queue->recording)) {
          const struct timespec delay = {.tv_sec = 0, .tv_nsec = 5000000L};
          nanosleep(&delay, NULL);
        }
        long long type_wait_ms = stt_now_ms() - type_wait_start_ms;
        if (type_wait_ms > 0) {
          LOG_DEBUG("run: capture=%llu segment=%u/%u type_wait elapsed_ms=%lld reason=recording_active\n",
                    job->capture_id, job->segment_index + 1, job->segment_count, type_wait_ms);
        }
        long long type_start_ms = stt_now_ms();
        int type_rc = stt_text_output_write(text, queue->config->type_delay_ms);
        type_ms = stt_now_ms() - type_start_ms;
        LOG_DEBUG("run: capture=%llu segment=%u/%u type elapsed_ms=%lld\n",
                  job->capture_id, job->segment_index + 1, job->segment_count, type_ms);
        if (type_rc == 0) {
          LOG_INFO("%s: %s\n", strcmp(stt_text_backend_name(), "stdout") == 0 ? "Printed" : "Typed", text);
        } else {
          LOG_WARN("Could not type the transcript; try --print to copy it manually.\n");
        }
      }
    } else if (rc != 0) {
      LOG_WARN("Could not transcribe this audio.\n");
    } else {
      LOG_INFO("No speech detected.\n");
    }
    free(text);
    stt_audio_buffer_free(&job->audio);
    long long total_ms = stt_now_ms() - job->release_ms;
    const char *longest = "post_roll";
    long long longest_ms = job->commit_ms;
    if (transcribe_ms > longest_ms) {
      longest = "transcribe";
      longest_ms = transcribe_ms;
    }
    long long queue_wait_ms = transcribe_start_ms - job->enqueue_ms;
    if (queue_wait_ms > longest_ms) {
      longest = "queue_wait";
      longest_ms = queue_wait_ms;
    }
    if (type_ms > longest_ms) {
      longest = "type";
      longest_ms = type_ms;
    }
    LOG_DEBUG("perf: capture=%llu segment=%u/%u total_release_to_done_ms=%lld longest=%s longest_ms=%lld post_roll_ms=%lld queue_wait_ms=%lld transcribe_ms=%lld type_ms=%lld queue_depth=%zu\n",
             job->capture_id, job->segment_index + 1, job->segment_count, total_ms, longest, longest_ms,
             job->commit_ms, queue_wait_ms, transcribe_ms, type_ms, depth_after_pop);
    LOG_DEBUG("run: capture=%llu segment=%u/%u done total_release_to_done_ms=%lld queue_depth=%zu\n",
              job->capture_id, job->segment_index + 1, job->segment_count, total_ms, depth_after_pop);
    free(job);
  }
}

static int transcribe_queue_start(TranscribeQueue *queue, const SttConfig *config, SttModel *model) {
  *queue = (TranscribeQueue){
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond = PTHREAD_COND_INITIALIZER,
    .config = config,
    .model = model,
  };
  atomic_init(&queue->recording, 0);
  int rc = pthread_create(&queue->thread, NULL, transcribe_worker, queue);
  if (rc != 0) {
    LOG_ERROR("run: transcribe_queue start failed rc=%d\n", rc);
    return -1;
  }
  queue->started = 1;
  LOG_DEBUG("run: transcribe_queue started workers=1\n");
  return 0;
}

static void transcribe_queue_stop(TranscribeQueue *queue) {
  if (!queue->started) return;
  pthread_mutex_lock(&queue->mutex);
  queue->stop = 1;
  pthread_cond_signal(&queue->cond);
  pthread_mutex_unlock(&queue->mutex);
  pthread_join(queue->thread, NULL);

  TranscribeJob *job = queue->head;
  while (job) {
    TranscribeJob *next = job->next;
    stt_audio_buffer_free(&job->audio);
    free(job);
    job = next;
  }
  pthread_cond_destroy(&queue->cond);
  pthread_mutex_destroy(&queue->mutex);
}

static int transcribe_queue_push(TranscribeQueue *queue,
                                 TranscribeJobKind kind,
                                 unsigned long long capture_id,
                                 unsigned int segment_index,
                                 unsigned int segment_count,
                                 SttAudioSegmentReason segment_reason,
                                 long long release_ms,
                                 long long commit_ms,
                                 SttAudioBuffer *audio,
                                 size_t *depth_out) {
  TranscribeJob *job = malloc(sizeof(*job));
  if (!job) return -1;
  job->audio = *audio;
  stt_audio_buffer_init(audio);
  job->kind = kind;
  job->capture_id = capture_id;
  job->segment_index = segment_index;
  job->segment_count = segment_count;
  job->segment_reason = segment_reason;
  job->release_ms = release_ms;
  job->commit_ms = commit_ms;
  job->enqueue_ms = stt_now_ms();
  job->next = NULL;

  pthread_mutex_lock(&queue->mutex);
  if (queue->tail) queue->tail->next = job;
  else queue->head = job;
  queue->tail = job;
  queue->depth++;
  size_t depth = queue->depth;
  pthread_cond_signal(&queue->cond);
  pthread_mutex_unlock(&queue->mutex);

  if (depth_out) *depth_out = depth;
  return 0;
}

static void on_hotkey(int pressed, void *user) {
  RunState *state = user;
  unsigned long long event_id = ++state->event_seq;
  size_t queue_depth = stt_log_enabled(STT_LOG_TRACE) ? transcribe_queue_depth(state->queue) : 0;
  LOG_TRACE("run: hotkey_event=%llu t_ms=%lld pressed=%d queue_depth=%zu recording=%d\n",
            event_id, stt_now_ms(), pressed, queue_depth, state->recording);
  if (pressed) {
    if (state->recording) {
      LOG_DEBUG("run: hotkey_event=%llu ignored pressed=1 reason=already_recording\n", event_id);
      return;
    }
    unsigned long long capture_id = ++state->capture_seq;
    int rc = stt_recorder_begin(state->recorder);
    LOG_DEBUG("run: capture=%llu begin rc=%d\n", capture_id, rc);
    if (rc == 0) {
      state->capture_begin_ms = stt_now_ms();
      state->recording = 1;
      atomic_store(&state->queue->recording, 1);
      LOG_INFO("Listening...\n");
    }
    return;
  }

  if (!state->recording) {
    LOG_DEBUG("run: hotkey_event=%llu ignored pressed=0 reason=not_recording\n", event_id);
    return;
  }
  unsigned long long capture_id = state->capture_seq;
  state->recording = 0;
  atomic_store(&state->queue->recording, 0);
  long long capture_start_ms = stt_now_ms();
  SttAudioSegment *segments = NULL;
  int commit_rc = stt_recorder_commit_segments(state->recorder, &segments);
  long long held_ms = capture_start_ms - state->capture_begin_ms;
  long long commit_ms = stt_now_ms() - capture_start_ms;
  size_t samples = 0;
  unsigned int segment_count = 0;
  for (SttAudioSegment *segment = segments; segment; segment = segment->next) {
    samples += segment->audio.len;
    segment_count++;
  }
  LOG_DEBUG("run: capture=%llu commit rc=%d elapsed_ms=%lld samples=%zu segments=%u\n",
            capture_id, commit_rc, commit_ms, samples, segment_count);

  size_t min_samples = (size_t)(STT_SAMPLE_RATE * (state->config->pre_roll_ms + 250) / 1000);
  if (commit_rc != 0 || samples <= min_samples) {
    if (commit_rc != 0) {
      LOG_WARN("Recording failed; no transcript was produced.\n");
    } else {
      LOG_INFO("Skipped %.1fs recording; it was too short.\n", samples_to_seconds(samples));
    }
    LOG_DEBUG("run: capture=%llu discarded reason=%s held_ms=%lld samples=%zu min_samples=%zu\n",
              capture_id, commit_rc != 0 ? "commit_failed" : "too_short_or_preroll_only", held_ms, samples, min_samples);
    stt_audio_segments_free(segments);
    return;
  }

  LOG_INFO("Transcribing %.1fs recording%s...\n",
           samples_to_seconds(samples), segment_count > 1 ? " in segments" : "");
  size_t depth = 0;
  for (SttAudioSegment *segment = segments; segment;) {
    SttAudioSegment *next = segment->next;
    segment->next = NULL;
    size_t segment_samples = segment->audio.len;
    const char *reason = segment->reason == STT_AUDIO_SEGMENT_MAX_AUDIO ? "max_audio" : "release";
    if (transcribe_queue_push(state->queue, TRANSCRIBE_JOB_HOTKEY, capture_id, segment->index, segment_count,
                              segment->reason, capture_start_ms, commit_ms, &segment->audio, &depth) != 0) {
      LOG_ERROR("run: capture=%llu segment=%u/%u discarded reason=queue_alloc_failed\n",
                capture_id, segment->index + 1, segment_count);
      stt_audio_buffer_free(&segment->audio);
      free(segment);
      stt_audio_segments_free(next);
      return;
    }
    LOG_DEBUG("run: capture=%llu segment=%u/%u queued reason=%s samples=%zu queue_depth=%zu\n",
              capture_id, segment->index + 1, segment_count, reason, segment_samples, depth);
    free(segment);
    segment = next;
  }
  LOG_DEBUG("run: capture=%llu queued elapsed_ms=%lld queue_depth=%zu recording=0\n",
            capture_id, stt_now_ms() - capture_start_ms, depth);
  LOG_DEBUG("perf: capture=%llu release_to_queue_ms=%lld commit_ms=%lld queue_depth=%zu\n",
           capture_id, stt_now_ms() - capture_start_ms, commit_ms, depth);
}

int main(int argc, char **argv) {
  SttConfig config;
  if (stt_parse_args(argc, argv, &config) != 0) {
    stt_print_usage(argv[0]);
    return 2;
  }

  SttModel *model = NULL;
  if (stt_model_load(&model, config.model_dir) != 0) return 1;
  SttRecorder *recorder = NULL;
  if (stt_recorder_open(&recorder, &config) != 0) {
    stt_model_free(model);
    return 1;
  }
  long long warmup_start_ms = stt_now_ms();
  LOG_INFO("Warming up speech model...\n");
  int warmup_rc = stt_transcribe_warmup(model, &config);
  LOG_DEBUG("run: warmup rc=%d elapsed_ms=%lld\n", warmup_rc, stt_now_ms() - warmup_start_ms);
  if (warmup_rc != 0) {
    stt_recorder_close(recorder);
    stt_model_free(model);
    return 1;
  }
  TranscribeQueue queue;
  if (transcribe_queue_start(&queue, &config, model) != 0) {
    stt_recorder_close(recorder);
    stt_model_free(model);
    return 1;
  }
  AutoLogState auto_log;
  if (auto_log_start(&auto_log, &config, &queue) != 0) {
    transcribe_queue_stop(&queue);
    stt_recorder_close(recorder);
    stt_model_free(model);
    return 1;
  }
  RunState state = {.config = &config, .recorder = recorder, .queue = &queue};
  int rc = stt_hotkey_loop(on_hotkey, &state);
  auto_log_stop(&auto_log);
  transcribe_queue_stop(&queue);
  stt_recorder_close(recorder);
  stt_model_free(model);
  return rc == 0 ? 0 : 1;
}
