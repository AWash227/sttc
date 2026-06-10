#ifndef STT_TDT_DECODE_H
#define STT_TDT_DECODE_H

#include <stddef.h>

typedef struct {
  int *ids;
  size_t len;
  size_t cap;
} SttTokenIds;

void stt_token_ids_init(SttTokenIds *ids);
void stt_token_ids_free(SttTokenIds *ids);
int stt_tdt_greedy_decode(const float *logits, size_t frames, size_t vocab_size, const int *durations, size_t duration_count, int blank_id, SttTokenIds *out);

#endif
