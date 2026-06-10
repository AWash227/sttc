#define _POSIX_C_SOURCE 200809L
#include "stt/audio.h"
#include "stt/cli.h"
#include "stt/cuda_runtime.h"
#include "stt/hotkey.h"
#include "stt/infer.h"
#include "stt/model.h"
#include "stt/model_install.h"
#include "stt/text_x11.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct TranscribeJob {
  SttAudioBuffer audio;
  unsigned long long capture_id;
  long long release_ms;
  long long enqueue_ms;
  struct TranscribeJob *next;
} TranscribeJob;

typedef struct {
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  pthread_t thread;
  TranscribeJob *head;
  TranscribeJob *tail;
  const SttOptions *opts;
  SttModel *model;
  atomic_int recording;
  size_t depth;
  int stop;
  int started;
} TranscribeQueue;

typedef struct {
  const SttOptions *opts;
  SttModel *model;
  SttRecorder *recorder;
  TranscribeQueue *queue;
  int recording;
  unsigned long long event_seq;
  unsigned long long capture_seq;
  long long capture_begin_ms;
} RunState;

static long long now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long long)ts.tv_sec * 1000 + ts.tv_nsec / 1000000;
}

static size_t transcribe_queue_depth(TranscribeQueue *queue) {
  pthread_mutex_lock(&queue->mutex);
  size_t depth = queue->depth;
  pthread_mutex_unlock(&queue->mutex);
  return depth;
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

    fprintf(stderr,
            "run: capture=%llu dequeue wait_ms=%lld queue_depth=%zu\n",
            job->capture_id, now_ms() - job->enqueue_ms, depth_after_pop);

    char *text = NULL;
    long long transcribe_start_ms = now_ms();
    int rc = stt_transcribe(queue->model, &job->audio, &text);
    fprintf(stderr, "run: capture=%llu transcribe rc=%d elapsed_ms=%lld\n",
            job->capture_id, rc, now_ms() - transcribe_start_ms);
    fprintf(stderr, "transcript: \"%s\"\n", text ? text : "");
    if (text && text[0]) {
      if (queue->opts->print_only || queue->opts->dry_run) {
        printf("%s\n", text);
        fflush(stdout);
      } else {
        long long type_wait_start_ms = now_ms();
        while (atomic_load(&queue->recording)) {
          const struct timespec delay = {.tv_sec = 0, .tv_nsec = 5000000L};
          nanosleep(&delay, NULL);
        }
        long long type_wait_ms = now_ms() - type_wait_start_ms;
        if (type_wait_ms > 0) {
          fprintf(stderr, "run: capture=%llu type_wait elapsed_ms=%lld reason=recording_active\n", job->capture_id, type_wait_ms);
        }
        long long type_start_ms = now_ms();
        stt_type_text_x11(text, queue->opts->type_delay_ms);
        fprintf(stderr, "run: capture=%llu type elapsed_ms=%lld\n", job->capture_id, now_ms() - type_start_ms);
      }
    } else if (rc != 0) {
      fprintf(stderr, "no transcript produced\n");
    }
    free(text);
    stt_audio_buffer_free(&job->audio);
    fprintf(stderr,
            "run: capture=%llu done total_release_to_done_ms=%lld queue_depth=%zu\n",
            job->capture_id, now_ms() - job->release_ms, depth_after_pop);
    free(job);
  }
}

static int transcribe_queue_start(TranscribeQueue *queue, const SttOptions *opts, SttModel *model) {
  *queue = (TranscribeQueue){
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .cond = PTHREAD_COND_INITIALIZER,
    .opts = opts,
    .model = model,
  };
  atomic_init(&queue->recording, 0);
  int rc = pthread_create(&queue->thread, NULL, transcribe_worker, queue);
  if (rc != 0) {
    fprintf(stderr, "run: transcribe_queue start failed rc=%d\n", rc);
    return -1;
  }
  queue->started = 1;
  fprintf(stderr, "run: transcribe_queue started workers=1\n");
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

static int transcribe_queue_push(TranscribeQueue *queue, unsigned long long capture_id, long long release_ms, SttAudioBuffer *audio, size_t *depth_out) {
  TranscribeJob *job = calloc(1, sizeof(*job));
  if (!job) return -1;
  job->audio = *audio;
  stt_audio_buffer_init(audio);
  job->capture_id = capture_id;
  job->release_ms = release_ms;
  job->enqueue_ms = now_ms();

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
  size_t queue_depth = transcribe_queue_depth(state->queue);
  fprintf(stderr, "run: hotkey_event=%llu t_ms=%lld pressed=%d queue_depth=%zu recording=%d\n",
          event_id, now_ms(), pressed, queue_depth, state->recording);
  if (pressed) {
    if (state->recording) {
      fprintf(stderr, "run: hotkey_event=%llu ignored pressed=1 reason=already_recording\n", event_id);
      return;
    }
    unsigned long long capture_id = ++state->capture_seq;
    int rc = stt_recorder_begin(state->recorder);
    fprintf(stderr, "run: capture=%llu begin rc=%d\n", capture_id, rc);
    if (rc == 0) {
      state->capture_begin_ms = now_ms();
      state->recording = 1;
      atomic_store(&state->queue->recording, 1);
    }
    return;
  }

  if (!state->recording) {
    fprintf(stderr, "run: hotkey_event=%llu ignored pressed=0 reason=not_recording\n", event_id);
    return;
  }
  unsigned long long capture_id = state->capture_seq;
  state->recording = 0;
  atomic_store(&state->queue->recording, 0);
  long long capture_start_ms = now_ms();
  SttAudioBuffer audio;
  stt_audio_buffer_init(&audio);
  int commit_rc = stt_recorder_commit(state->recorder, &audio);
  long long held_ms = capture_start_ms - state->capture_begin_ms;
  fprintf(stderr, "run: capture=%llu commit rc=%d elapsed_ms=%lld samples=%zu\n",
          capture_id, commit_rc, now_ms() - capture_start_ms, audio.len);

  size_t min_samples = (size_t)(16000 * (state->opts->pre_roll_ms + 250) / 1000);
  if (commit_rc != 0 || audio.len <= min_samples) {
    fprintf(stderr,
            "run: capture=%llu discarded reason=%s held_ms=%lld samples=%zu min_samples=%zu\n",
            capture_id, commit_rc != 0 ? "commit_failed" : "too_short_or_preroll_only", held_ms, audio.len, min_samples);
    stt_audio_buffer_free(&audio);
    return;
  }

  size_t depth = 0;
  if (transcribe_queue_push(state->queue, capture_id, capture_start_ms, &audio, &depth) != 0) {
    fprintf(stderr, "run: capture=%llu discarded reason=queue_alloc_failed\n", capture_id);
    stt_audio_buffer_free(&audio);
    return;
  }
  fprintf(stderr, "run: capture=%llu queued elapsed_ms=%lld queue_depth=%zu recording=0\n",
          capture_id, now_ms() - capture_start_ms, depth);
}

static int cmd_test_gpu(void) {
  SttCuda cuda;
  if (stt_cuda_init(&cuda) != 0) {
    fprintf(stderr, "CUDA test failed: %s\n", stt_cuda_last_error());
    return 1;
  }
  printf("CUDA device: %s\n", cuda.name);
  printf("VRAM: %.2f GiB\n", (double)cuda.total_mem / (1024.0 * 1024.0 * 1024.0));
  stt_cuda_destroy(&cuda);
  return 0;
}

int main(int argc, char **argv) {
  SttOptions opts;
  if (stt_parse_args(argc, argv, &opts) != 0) {
    stt_print_usage(argv[0]);
    return 2;
  }

  switch (opts.command) {
    case STT_CMD_INSTALL_MODEL:
      return stt_install_model_v3(opts.model_dir) == 0 ? 0 : 1;
    case STT_CMD_INSPECT_MODEL:
      return stt_model_inspect(opts.model_dir) == 0 ? 0 : 1;
    case STT_CMD_TEST_GPU:
      return cmd_test_gpu();
    case STT_CMD_RUN: {
      SttModel *model = NULL;
      if (stt_model_load(&model, opts.model_dir) != 0) return 1;
      SttRecorder *recorder = NULL;
      if (stt_recorder_open(&recorder, 16000, opts.max_audio_sec, opts.pre_roll_ms, opts.post_roll_ms) != 0) {
        stt_model_free(model);
        return 1;
      }
      long long warmup_start_ms = now_ms();
      int warmup_rc = stt_transcribe_warmup(model);
      fprintf(stderr, "run: warmup rc=%d elapsed_ms=%lld\n", warmup_rc, now_ms() - warmup_start_ms);
      if (warmup_rc != 0) {
        stt_recorder_close(recorder);
        stt_model_free(model);
        return 1;
      }
      TranscribeQueue queue;
      if (transcribe_queue_start(&queue, &opts, model) != 0) {
        stt_recorder_close(recorder);
        stt_model_free(model);
        return 1;
      }
      RunState state = {.opts = &opts, .model = model, .recorder = recorder, .queue = &queue};
      int rc = stt_hotkey_loop(opts.hotkey, on_hotkey, &state);
      transcribe_queue_stop(&queue);
      stt_recorder_close(recorder);
      stt_model_free(model);
      return rc == 0 ? 0 : 1;
    }
    default:
      stt_print_usage(argv[0]);
      return 2;
  }
}
