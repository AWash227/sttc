#define _POSIX_C_SOURCE 200809L
#include "stt/tdt_decode.h"

#include <float.h>
#include <stdlib.h>
#include <string.h>

void stt_token_ids_init(SttTokenIds *ids) {
  memset(ids, 0, sizeof(*ids));
}

void stt_token_ids_free(SttTokenIds *ids) {
  free(ids->ids);
  memset(ids, 0, sizeof(*ids));
}

static int push_id(SttTokenIds *out, int id) {
  if (out->len == out->cap) {
    size_t next_cap = out->cap ? out->cap * 2 : 128;
    int *next = realloc(out->ids, next_cap * sizeof(*next));
    if (!next) return -1;
    out->ids = next;
    out->cap = next_cap;
  }
  out->ids[out->len++] = id;
  return 0;
}

static int argmax(const float *xs, size_t n) {
  int best = 0;
  float best_v = -FLT_MAX;
  for (size_t i = 0; i < n; ++i) {
    if (xs[i] > best_v) {
      best_v = xs[i];
      best = (int)i;
    }
  }
  return best;
}

int stt_tdt_greedy_decode(const float *logits, size_t frames, size_t vocab_size, const int *durations, size_t duration_count, int blank_id, SttTokenIds *out) {
  size_t stride = vocab_size + duration_count;
  size_t t = 0;
  while (t < frames) {
    const float *frame = logits + t * stride;
    int token = argmax(frame, vocab_size);
    int dur_idx = argmax(frame + vocab_size, duration_count);
    int duration = durations[dur_idx];
    if (token == blank_id) {
      if (duration < 1) duration = 1;
    } else {
      if (push_id(out, token) != 0) return -1;
    }
    if (duration < 0) duration = 0;
    if (duration == 0 && token == blank_id) duration = 1;
    t += (size_t)duration;
    if (duration == 0) ++t;
  }
  return 0;
}
