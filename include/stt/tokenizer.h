#ifndef STT_TOKENIZER_H
#define STT_TOKENIZER_H

#include <stddef.h>

typedef struct {
  char **tokens;
  size_t count;
} SttTokenizer;

int stt_tokenizer_load(SttTokenizer *tok, const char *path);
void stt_tokenizer_free(SttTokenizer *tok);
int stt_tokenizer_decode(const SttTokenizer *tok, const int *ids, size_t len, char **text_out);

#endif
