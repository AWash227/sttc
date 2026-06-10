#define _POSIX_C_SOURCE 200809L
#include "stt/tokenizer.h"
#include "stt/fs.h"

#include <json-c/json.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ensure_token(SttTokenizer *tok, size_t id) {
  if (id < tok->count) return 0;
  size_t next_count = id + 1;
  char **next = realloc(tok->tokens, next_count * sizeof(*tok->tokens));
  if (!next) return -1;
  for (size_t i = tok->count; i < next_count; ++i) next[i] = NULL;
  tok->tokens = next;
  tok->count = next_count;
  return 0;
}

int stt_tokenizer_load(SttTokenizer *tok, const char *path) {
  memset(tok, 0, sizeof(*tok));
  size_t len = 0;
  char *json = stt_read_file(path, &len);
  (void)len;
  if (!json) {
    perror(path);
    return -1;
  }

  json_object *root = json_tokener_parse(json);
  free(json);
  if (!root) return -1;

  json_object *model = NULL, *vocab = NULL;
  if (!json_object_object_get_ex(root, "model", &model) ||
      !json_object_object_get_ex(model, "vocab", &vocab) ||
      !json_object_is_type(vocab, json_type_object)) {
    json_object_put(root);
    return -1;
  }

  json_object_object_foreach(vocab, token, id_o) {
    int id = json_object_get_int(id_o);
    if (id < 0) continue;
    if (ensure_token(tok, (size_t)id) != 0) {
      json_object_put(root);
      return -1;
    }
    tok->tokens[id] = strdup(token);
    if (!tok->tokens[id]) {
      json_object_put(root);
      return -1;
    }
  }

  json_object_put(root);
  return 0;
}

void stt_tokenizer_free(SttTokenizer *tok) {
  if (!tok) return;
  for (size_t i = 0; i < tok->count; ++i) free(tok->tokens[i]);
  free(tok->tokens);
  memset(tok, 0, sizeof(*tok));
}

static int is_special(const char *s) {
  return s[0] == '<' && strstr(s, "|") != NULL;
}

static int append_bytes(char **buf, size_t *len, size_t *cap, const char *s, size_t n) {
  if (*len + n + 1 > *cap) {
    size_t next_cap = *cap ? *cap : 256;
    while (*len + n + 1 > next_cap) next_cap *= 2;
    char *next = realloc(*buf, next_cap);
    if (!next) return -1;
    *buf = next;
    *cap = next_cap;
  }
  memcpy(*buf + *len, s, n);
  *len += n;
  (*buf)[*len] = '\0';
  return 0;
}

int stt_tokenizer_decode(const SttTokenizer *tok, const int *ids, size_t id_len, char **text_out) {
  char *out = NULL;
  size_t len = 0, cap = 0;
  for (size_t i = 0; i < id_len; ++i) {
    int id = ids[i];
    if (id < 0 || (size_t)id >= tok->count || !tok->tokens[id]) continue;
    const char *piece = tok->tokens[id];
    if (is_special(piece)) continue;

    for (const char *p = piece; *p;) {
      if ((unsigned char)p[0] == 0xe2 && (unsigned char)p[1] == 0x96 && (unsigned char)p[2] == 0x81) {
        if (append_bytes(&out, &len, &cap, " ", 1) != 0) return -1;
        p += 3;
      } else {
        size_t n = 1;
        if (((unsigned char)*p & 0xe0) == 0xc0) n = 2;
        else if (((unsigned char)*p & 0xf0) == 0xe0) n = 3;
        else if (((unsigned char)*p & 0xf8) == 0xf0) n = 4;
        if (append_bytes(&out, &len, &cap, p, n) != 0) return -1;
        p += n;
      }
    }
  }

  while (len > 0 && out[0] == ' ') {
    memmove(out, out + 1, len);
    --len;
  }
  if (!out) out = strdup("");
  *text_out = out;
  return 0;
}
