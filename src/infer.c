#define _POSIX_C_SOURCE 200809L
#include "stt/infer.h"
#include "stt/features.h"
#include "stt/fs.h"
#include "stt/log.h"

#include <cuda.h>
#include <onnxruntime_c_api.h>
#include <errno.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TDT_FEATURE_BINS 128
#define TDT_ENCODER_DIM 1024
#define TDT_STATE_LAYERS 2
#define TDT_STATE_DIM 640
#define TDT_VOCAB_SIZE 1025
#define TDT_DURATION_COUNT 5
#define TDT_BLANK_ID 1024
#define TDT_MAX_TOKENS_PER_STEP 10

typedef struct {
  char **tokens;
  size_t count;
} TdtVocab;

typedef struct {
  const OrtApi *api;
  OrtEnv *env;
  OrtSessionOptions *options;
  OrtSession *encoder;
  OrtSession *decoder;
  OrtMemoryInfo *memory;
} OrtTdt;

typedef struct {
  int64_t decoder_calls;
  int64_t blank_steps;
  int64_t emitted_tokens;
  int64_t advanced_frames;
  long long decoder_ms;
} DecodeStats;

typedef struct {
  char *model_dir;
  char *variant;
  TdtVocab vocab;
  OrtTdt ort;
  int ready;
} TdtRuntime;

static TdtRuntime g_tdt_runtime;

static double elapsed_rtf(long long elapsed_ms, double audio_sec) {
  return audio_sec > 0.0 ? (double)elapsed_ms / (audio_sec * 1000.0) : 0.0;
}

static const char *requested_tdt_variant(void) {
  const char *variant = getenv("STT_TDT_VARIANT");
  if (!variant || !*variant || strcmp(variant, "auto") == 0) return "fp32";
  if (strcmp(variant, "fp32") == 0 || strcmp(variant, "int8") == 0) return variant;
  LOG_WARN("infer: unknown STT_TDT_VARIANT=%s using=fp32\n", variant);
  return "fp32";
}

static int file_exists_in_dir(const char *dir, const char *name) {
  char *path = stt_path_join(dir, name);
  if (!path) return 0;
  int exists = stt_file_exists(path);
  free(path);
  return exists;
}

static void tdt_variant_files(const char *model_dir, const char **variant_out, const char **encoder_file_out, const char **decoder_file_out) {
  const char *variant = requested_tdt_variant();
  const char *encoder_file = "encoder-model.onnx";
  const char *decoder_file = "decoder_joint-model.onnx";

  if (strcmp(variant, "int8") == 0) {
    if (file_exists_in_dir(model_dir, "encoder-model.int8.onnx") &&
        file_exists_in_dir(model_dir, "decoder_joint-model.int8.onnx")) {
      encoder_file = "encoder-model.int8.onnx";
      decoder_file = "decoder_joint-model.int8.onnx";
    } else {
      LOG_WARN("infer: tdt_variant=int8 unavailable; falling back to fp32\n");
      variant = "fp32";
    }
  }

  *variant_out = variant;
  *encoder_file_out = encoder_file;
  *decoder_file_out = decoder_file;
}

static OrtCudnnConvAlgoSearch cudnn_algo_search_from_env(const char **name_out) {
  const char *value = getenv("STT_CUDNN_CONV_ALGO_SEARCH");
  if (!value || !*value || strcmp(value, "DEFAULT") == 0) {
    *name_out = "DEFAULT";
    return OrtCudnnConvAlgoSearchDefault;
  }
  if (strcmp(value, "HEURISTIC") == 0) {
    *name_out = "HEURISTIC";
    return OrtCudnnConvAlgoSearchHeuristic;
  }
  if (strcmp(value, "EXHAUSTIVE") == 0) {
    *name_out = "EXHAUSTIVE";
    return OrtCudnnConvAlgoSearchExhaustive;
  }
  LOG_WARN("infer: unknown STT_CUDNN_CONV_ALGO_SEARCH=%s using=DEFAULT\n", value);
  *name_out = "DEFAULT";
  return OrtCudnnConvAlgoSearchDefault;
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

static int ort_check(OrtTdt *ort, OrtStatus *status, const char *what) {
  if (!status) return 0;
  LOG_ERROR("%s: %s\n", what, ort->api->GetErrorMessage(status));
  ort->api->ReleaseStatus(status);
  return -1;
}

static int cuda_device_available(void) {
  int count = 0;
  if (cuInit(0) != CUDA_SUCCESS) return 0;
  if (cuDeviceGetCount(&count) != CUDA_SUCCESS) return 0;
  return count > 0;
}

static void tdt_vocab_free(TdtVocab *vocab) {
  if (!vocab) return;
  for (size_t i = 0; i < vocab->count; ++i) free(vocab->tokens[i]);
  free(vocab->tokens);
  memset(vocab, 0, sizeof(*vocab));
}

static int tdt_vocab_load(TdtVocab *vocab, const char *path) {
  memset(vocab, 0, sizeof(*vocab));
  size_t len = 0;
  char *text = stt_read_file(path, &len);
  if (!text) {
    LOG_ERROR("failed to read vocab: %s: %s\n", path, strerror(errno));
    return -1;
  }

  vocab->tokens = calloc(TDT_VOCAB_SIZE, sizeof(*vocab->tokens));
  if (!vocab->tokens) {
    free(text);
    return -1;
  }
  vocab->count = TDT_VOCAB_SIZE;

  char *save = NULL;
  for (char *line = strtok_r(text, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
    char *sep = strrchr(line, ' ');
    if (!sep) continue;
    *sep++ = '\0';
    char *end = NULL;
    long id = strtol(sep, &end, 10);
    if (*line && id >= 0 && id < TDT_VOCAB_SIZE && end && *end == '\0') {
      vocab->tokens[id] = strdup(line);
      if (!vocab->tokens[id]) {
        free(text);
        tdt_vocab_free(vocab);
        return -1;
      }
    }
  }

  free(text);
  return vocab->tokens[TDT_BLANK_ID] ? 0 : -1;
}

static int tdt_vocab_decode(const TdtVocab *vocab, const int *ids, size_t id_count, char **text_out) {
  char *out = NULL;
  size_t len = 0;
  size_t cap = 0;

  for (size_t i = 0; i < id_count; ++i) {
    int id = ids[i];
    if (id < 0 || (size_t)id >= vocab->count || !vocab->tokens[id]) continue;
    const char *piece = vocab->tokens[id];
    if (piece[0] == '<') continue;

    for (const char *p = piece; *p;) {
      if ((unsigned char)p[0] == 0xe2 && (unsigned char)p[1] == 0x96 && (unsigned char)p[2] == 0x81) {
        if (append_bytes(&out, &len, &cap, " ", 1) != 0) goto fail;
        p += 3;
      } else {
        size_t n = 1;
        if (((unsigned char)*p & 0xe0) == 0xc0) n = 2;
        else if (((unsigned char)*p & 0xf0) == 0xe0) n = 3;
        else if (((unsigned char)*p & 0xf8) == 0xf0) n = 4;
        if (append_bytes(&out, &len, &cap, p, n) != 0) goto fail;
        p += n;
      }
    }
  }

  while (len > 0 && out[0] == ' ') {
    memmove(out, out + 1, len);
    --len;
  }
  if (!out) out = strdup("");
  if (!out) return -1;
  *text_out = out;
  return 0;

fail:
  free(out);
  return -1;
}

static void ort_tdt_free(OrtTdt *ort) {
  if (!ort || !ort->api) return;
  if (ort->memory) ort->api->ReleaseMemoryInfo(ort->memory);
  if (ort->decoder) ort->api->ReleaseSession(ort->decoder);
  if (ort->encoder) ort->api->ReleaseSession(ort->encoder);
  if (ort->options) ort->api->ReleaseSessionOptions(ort->options);
  if (ort->env) ort->api->ReleaseEnv(ort->env);
  memset(ort, 0, sizeof(*ort));
}

static void tdt_runtime_free(TdtRuntime *runtime) {
  if (!runtime) return;
  ort_tdt_free(&runtime->ort);
  tdt_vocab_free(&runtime->vocab);
  free(runtime->model_dir);
  free(runtime->variant);
  memset(runtime, 0, sizeof(*runtime));
}

static int ort_tdt_init(OrtTdt *ort, const char *model_dir, const char *variant, const char *encoder_file, const char *decoder_file) {
  long long start_ms = stt_now_ms();
  memset(ort, 0, sizeof(*ort));
  ort->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  char *encoder_path = stt_path_join(model_dir, encoder_file);
  char *decoder_path = stt_path_join(model_dir, decoder_file);
  if (!encoder_path || !decoder_path) goto fail;
  LOG_INFO("infer: tdt_variant=%s encoder_file=%s decoder_file=%s\n", variant, encoder_file, decoder_file);

  if (ort_check(ort, ort->api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "stt", &ort->env), "CreateEnv") != 0) goto fail;
  if (ort_check(ort, ort->api->CreateSessionOptions(&ort->options), "CreateSessionOptions") != 0) goto fail;
  ort_check(ort, ort->api->SetIntraOpNumThreads(ort->options, 1), "SetIntraOpNumThreads");
  ort_check(ort, ort->api->SetSessionGraphOptimizationLevel(ort->options, ORT_ENABLE_ALL), "SetSessionGraphOptimizationLevel");

  int cuda_ep = 0;
  if (!cuda_device_available()) {
    LOG_INFO("infer: ort_init cuda_ep_configured=0 reason=\"no CUDA device available\"\n");
  } else {
    const char *algo_name = NULL;
    OrtCudnnConvAlgoSearch algo_search = cudnn_algo_search_from_env(&algo_name);
    OrtCUDAProviderOptions cuda_options = {
      .device_id = 0,
      .cudnn_conv_algo_search = algo_search,
      .gpu_mem_limit = SIZE_MAX,
      .arena_extend_strategy = 1,
      .do_copy_in_default_stream = 1,
    };
    OrtStatus *cuda_status = ort->api->SessionOptionsAppendExecutionProvider_CUDA(ort->options, &cuda_options);
    if (cuda_status) {
      LOG_WARN("infer: ort_init cuda_ep_configured=0 reason=\"%s\"\n", ort->api->GetErrorMessage(cuda_status));
      ort->api->ReleaseStatus(cuda_status);
    } else {
      cuda_ep = 1;
      LOG_INFO("infer: ort_init cuda_ep_configured=1 cudnn_conv_algo_search=%s\n", algo_name);
    }
  }
  if (ort_check(ort, ort->api->CreateSession(ort->env, encoder_path, ort->options, &ort->encoder), "CreateSession encoder") != 0) goto fail;
  if (ort_check(ort, ort->api->CreateSession(ort->env, decoder_path, ort->options, &ort->decoder), "CreateSession decoder") != 0) goto fail;
  if (ort_check(ort, ort->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort->memory), "CreateCpuMemoryInfo") != 0) goto fail;
  LOG_INFO("infer: ort_init done elapsed_ms=%lld cuda_ep=%d\n", stt_now_ms() - start_ms, cuda_ep);

  free(encoder_path);
  free(decoder_path);
  return 0;

fail:
  free(encoder_path);
  free(decoder_path);
  ort_tdt_free(ort);
  return -1;
}

static int tdt_runtime_get(const char *model_dir, TdtRuntime **runtime_out) {
  long long start_ms = stt_now_ms();
  const char *variant = NULL;
  const char *encoder_file = NULL;
  const char *decoder_file = NULL;
  tdt_variant_files(model_dir, &variant, &encoder_file, &decoder_file);

  if (g_tdt_runtime.ready &&
      strcmp(g_tdt_runtime.model_dir, model_dir) == 0 &&
      strcmp(g_tdt_runtime.variant, variant) == 0) {
    LOG_TRACE("infer: runtime_cache hit=1 elapsed_ms=0\n");
    *runtime_out = &g_tdt_runtime;
    return 0;
  }

  LOG_INFO("infer: runtime_cache hit=0 loading model_dir=%s variant=%s\n", model_dir, variant);
  tdt_runtime_free(&g_tdt_runtime);
  char *vocab_path = stt_path_join(model_dir, "vocab.txt");
  if (!vocab_path) return -1;
  g_tdt_runtime.model_dir = strdup(model_dir);
  g_tdt_runtime.variant = strdup(variant);
  if (!g_tdt_runtime.model_dir || !g_tdt_runtime.variant) {
    free(vocab_path);
    tdt_runtime_free(&g_tdt_runtime);
    return -1;
  }
  if (tdt_vocab_load(&g_tdt_runtime.vocab, vocab_path) != 0) {
    free(vocab_path);
    tdt_runtime_free(&g_tdt_runtime);
    return -1;
  }
  free(vocab_path);
  if (ort_tdt_init(&g_tdt_runtime.ort, model_dir, variant, encoder_file, decoder_file) != 0) {
    tdt_runtime_free(&g_tdt_runtime);
    return -1;
  }
  g_tdt_runtime.ready = 1;
  LOG_INFO("infer: runtime_cache loaded elapsed_ms=%lld tokens=%zu variant=%s\n",
           stt_now_ms() - start_ms, g_tdt_runtime.vocab.count, g_tdt_runtime.variant);
  *runtime_out = &g_tdt_runtime;
  return 0;
}

static int make_tensor(OrtTdt *ort, void *data, size_t bytes, const int64_t *shape, size_t rank, ONNXTensorElementDataType type, OrtValue **out) {
  return ort_check(ort,
                   ort->api->CreateTensorWithDataAsOrtValue(ort->memory, data, bytes, shape, rank, type, out),
                   "CreateTensorWithDataAsOrtValue");
}

static int get_tensor_data(OrtTdt *ort, OrtValue *value, float **data_out) {
  return ort_check(ort, ort->api->GetTensorMutableData(value, (void **)data_out), "GetTensorMutableData");
}

static int get_tensor_i64(OrtTdt *ort, OrtValue *value, int64_t **data_out) {
  return ort_check(ort, ort->api->GetTensorMutableData(value, (void **)data_out), "GetTensorMutableData");
}

static int encoder_run(OrtTdt *ort, const SttFeatures *features, OrtValue **encoded_out, int64_t *encoded_len_out, long long *elapsed_ms_out) {
  long long start_ms = stt_now_ms();
  int64_t length = (int64_t)features->valid_frames;
  int64_t input_shape[] = {1, TDT_FEATURE_BINS, (int64_t)features->valid_frames};
  int64_t length_shape[] = {1};
  OrtValue *inputs[2] = {NULL, NULL};
  OrtValue *outputs[2] = {NULL, NULL};
  const char *input_names[] = {"audio_signal", "length"};
  const char *output_names[] = {"outputs", "encoded_lengths"};
  int rc = -1;

  if (make_tensor(ort, features->data, features->valid_frames * TDT_FEATURE_BINS * sizeof(*features->data), input_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[0]) != 0) goto done;
  if (make_tensor(ort, &length, sizeof(length), length_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &inputs[1]) != 0) goto done;
  if (ort_check(ort, ort->api->Run(ort->encoder, NULL, input_names, (const OrtValue *const *)inputs, 2, output_names, 2, outputs), "Run encoder") != 0) goto done;

  int64_t *encoded_lengths = NULL;
  if (get_tensor_i64(ort, outputs[1], &encoded_lengths) != 0) goto done;
  *encoded_out = outputs[0];
  *encoded_len_out = encoded_lengths[0];
  outputs[0] = NULL;
  rc = 0;

done:
  if (outputs[0]) ort->api->ReleaseValue(outputs[0]);
  if (outputs[1]) ort->api->ReleaseValue(outputs[1]);
  if (inputs[0]) ort->api->ReleaseValue(inputs[0]);
  if (inputs[1]) ort->api->ReleaseValue(inputs[1]);
  if (elapsed_ms_out) *elapsed_ms_out = stt_now_ms() - start_ms;
  return rc;
}

static int push_id(int **ids, size_t *len, size_t *cap, int id) {
  if (*len == *cap) {
    size_t next_cap = *cap ? *cap * 2 : 128;
    int *next = realloc(*ids, next_cap * sizeof(*next));
    if (!next) return -1;
    *ids = next;
    *cap = next_cap;
  }
  (*ids)[(*len)++] = id;
  return 0;
}

static int decoder_step(OrtTdt *ort,
                        const float *encoder,
                        int64_t encoded_len,
                        int64_t frame,
                        int32_t target,
                        const float *state1,
                        const float *state2,
                        float *next_state1,
                        float *next_state2,
                        int *token_out,
                        int *duration_out) {
  float encoder_slice[TDT_ENCODER_DIM];
  for (size_t d = 0; d < TDT_ENCODER_DIM; ++d) encoder_slice[d] = encoder[d * encoded_len + frame];

  int32_t target_length = 1;
  int64_t encoder_shape[] = {1, TDT_ENCODER_DIM, 1};
  int64_t target_shape[] = {1, 1};
  int64_t target_length_shape[] = {1};
  int64_t state_shape[] = {TDT_STATE_LAYERS, 1, TDT_STATE_DIM};
  OrtValue *inputs[5] = {NULL, NULL, NULL, NULL, NULL};
  OrtValue *outputs[4] = {NULL, NULL, NULL, NULL};
  const char *input_names[] = {"encoder_outputs", "targets", "target_length", "input_states_1", "input_states_2"};
  const char *output_names[] = {"outputs", "prednet_lengths", "output_states_1", "output_states_2"};
  int rc = -1;

  if (make_tensor(ort, encoder_slice, sizeof(encoder_slice), encoder_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[0]) != 0) goto done;
  if (make_tensor(ort, &target, sizeof(target), target_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &inputs[1]) != 0) goto done;
  if (make_tensor(ort, &target_length, sizeof(target_length), target_length_shape, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, &inputs[2]) != 0) goto done;
  if (make_tensor(ort, (void *)state1, TDT_STATE_LAYERS * TDT_STATE_DIM * sizeof(*state1), state_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[3]) != 0) goto done;
  if (make_tensor(ort, (void *)state2, TDT_STATE_LAYERS * TDT_STATE_DIM * sizeof(*state2), state_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputs[4]) != 0) goto done;
  if (ort_check(ort, ort->api->Run(ort->decoder, NULL, input_names, (const OrtValue *const *)inputs, 5, output_names, 4, outputs), "Run decoder") != 0) goto done;

  float *logits = NULL;
  if (get_tensor_data(ort, outputs[0], &logits) != 0) goto done;
  int token = argmax(logits, TDT_VOCAB_SIZE);
  int duration = argmax(logits + TDT_VOCAB_SIZE, TDT_DURATION_COUNT);
  if (token != TDT_BLANK_ID) {
    float *out_state1 = NULL;
    float *out_state2 = NULL;
    if (get_tensor_data(ort, outputs[2], &out_state1) != 0) goto done;
    if (get_tensor_data(ort, outputs[3], &out_state2) != 0) goto done;
    memcpy(next_state1, out_state1, TDT_STATE_LAYERS * TDT_STATE_DIM * sizeof(*next_state1));
    memcpy(next_state2, out_state2, TDT_STATE_LAYERS * TDT_STATE_DIM * sizeof(*next_state2));
  }
  *token_out = token;
  *duration_out = duration;
  rc = 0;

done:
  for (size_t i = 0; i < 5; ++i) if (inputs[i]) ort->api->ReleaseValue(inputs[i]);
  if (outputs[0]) ort->api->ReleaseValue(outputs[0]);
  if (outputs[1]) ort->api->ReleaseValue(outputs[1]);
  if (outputs[2]) ort->api->ReleaseValue(outputs[2]);
  if (outputs[3]) ort->api->ReleaseValue(outputs[3]);
  return rc;
}

static int tdt_greedy_decode_onnx(OrtTdt *ort, OrtValue *encoded_value, int64_t encoded_len, int **ids_out, size_t *id_count_out, DecodeStats *stats) {
  float *encoder = NULL;
  if (get_tensor_data(ort, encoded_value, &encoder) != 0) return -1;

  float state1[TDT_STATE_LAYERS * TDT_STATE_DIM] = {0};
  float state2[TDT_STATE_LAYERS * TDT_STATE_DIM] = {0};
  float next_state1[TDT_STATE_LAYERS * TDT_STATE_DIM];
  float next_state2[TDT_STATE_LAYERS * TDT_STATE_DIM];
  int *ids = NULL;
  size_t id_count = 0;
  size_t id_cap = 0;
  int32_t target = TDT_BLANK_ID;
  int emitted_tokens = 0;
  memset(stats, 0, sizeof(*stats));

  for (int64_t t = 0; t < encoded_len;) {
    int token = TDT_BLANK_ID;
    int duration = 0;
    long long step_start_ms = stt_now_ms();
    if (decoder_step(ort, encoder, encoded_len, t, target, state1, state2, next_state1, next_state2, &token, &duration) != 0) goto fail;
    stats->decoder_ms += stt_now_ms() - step_start_ms;
    stats->decoder_calls++;

    if (token != TDT_BLANK_ID) {
      if (push_id(&ids, &id_count, &id_cap, token) != 0) goto fail;
      stats->emitted_tokens++;
      memcpy(state1, next_state1, sizeof(state1));
      memcpy(state2, next_state2, sizeof(state2));
      target = token;
      emitted_tokens++;
    }

    if (token != TDT_BLANK_ID && duration > 0) {
      t += duration;
      stats->advanced_frames += duration;
      emitted_tokens = 0;
    } else if (token == TDT_BLANK_ID || emitted_tokens == TDT_MAX_TOKENS_PER_STEP) {
      t++;
      stats->advanced_frames++;
      if (token == TDT_BLANK_ID) stats->blank_steps++;
      emitted_tokens = 0;
    }
  }

  *ids_out = ids;
  *id_count_out = id_count;
  return 0;

fail:
  free(ids);
  return -1;
}

static int transcribe_tdt_onnx(SttModel *model, const SttAudioBuffer *audio, char **text_out) {
  SttFeatures features;
  long long total_start_ms = stt_now_ms();
  double audio_sec = audio->sample_rate ? (double)audio->len / (double)audio->sample_rate : 0.0;
  long long feature_start_ms = stt_now_ms();
  if (stt_extract_features(audio, &features) != 0) {
    LOG_ERROR("feature extraction failed\n");
    *text_out = strdup("");
    return -1;
  }
  long long feature_ms = stt_now_ms() - feature_start_ms;
  LOG_TRACE("infer: features elapsed_ms=%lld frames=%zu valid=%zu bins=%zu\n",
            feature_ms, features.frames, features.valid_frames, features.bins);

  TdtRuntime *runtime = NULL;
  OrtValue *encoded = NULL;
  int *ids = NULL;
  size_t id_count = 0;
  int64_t encoded_len = 0;
  long long runtime_ms = 0;
  long long encoder_ms = 0;
  long long text_ms = 0;
  DecodeStats decode_stats = {0};
  int rc = -1;

  long long runtime_start_ms = stt_now_ms();
  if (tdt_runtime_get(stt_model_dir(model), &runtime) != 0) goto done;
  runtime_ms = stt_now_ms() - runtime_start_ms;
  if (encoder_run(&runtime->ort, &features, &encoded, &encoded_len, &encoder_ms) != 0) goto done;
  LOG_TRACE("infer: encoder elapsed_ms=%lld encoded_frames=%lld\n", encoder_ms, (long long)encoded_len);
  if (tdt_greedy_decode_onnx(&runtime->ort, encoded, encoded_len, &ids, &id_count, &decode_stats) != 0) goto done;
  LOG_TRACE("infer: decoder elapsed_ms=%lld calls=%lld blank_steps=%lld emitted_tokens=%lld advanced_frames=%lld output_ids=%zu\n",
            decode_stats.decoder_ms,
            (long long)decode_stats.decoder_calls,
            (long long)decode_stats.blank_steps,
            (long long)decode_stats.emitted_tokens,
            (long long)decode_stats.advanced_frames,
            id_count);
  long long text_start_ms = stt_now_ms();
  if (tdt_vocab_decode(&runtime->vocab, ids, id_count, text_out) != 0) goto done;
  text_ms = stt_now_ms() - text_start_ms;
  LOG_TRACE("infer: text_decode elapsed_ms=%lld chars=%zu\n", text_ms, *text_out ? strlen(*text_out) : 0);
  rc = 0;

done:
  if (rc != 0) *text_out = strdup("");
  free(ids);
  if (encoded && runtime) runtime->ort.api->ReleaseValue(encoded);
  stt_features_free(&features);
  long long total_ms = stt_now_ms() - total_start_ms;
  const char *longest = "features";
  long long longest_ms = feature_ms;
  if (runtime_ms > longest_ms) {
    longest = "runtime";
    longest_ms = runtime_ms;
  }
  if (encoder_ms > longest_ms) {
    longest = "encoder";
    longest_ms = encoder_ms;
  }
  if (decode_stats.decoder_ms > longest_ms) {
    longest = "decoder";
    longest_ms = decode_stats.decoder_ms;
  }
  if (text_ms > longest_ms) {
    longest = "text_decode";
    longest_ms = text_ms;
  }
  long long measured_ms = feature_ms + runtime_ms + encoder_ms + decode_stats.decoder_ms + text_ms;
  long long overhead_ms = total_ms > measured_ms ? total_ms - measured_ms : 0;
  LOG_INFO("perf: infer total_ms=%lld rtf=%.3f longest=%s longest_ms=%lld features_ms=%lld runtime_ms=%lld encoder_ms=%lld decoder_ms=%lld text_ms=%lld overhead_ms=%lld rc=%d\n",
           total_ms,
           elapsed_rtf(total_ms, audio_sec),
           longest,
           longest_ms,
           feature_ms,
           runtime_ms,
           encoder_ms,
           decode_stats.decoder_ms,
           text_ms,
           overhead_ms,
           rc);
  LOG_TRACE("infer: total elapsed_ms=%lld rtf=%.3f rc=%d\n", total_ms, elapsed_rtf(total_ms, audio_sec), rc);
  return rc;
}

static int tdt_runtime_prime_duration(TdtRuntime *runtime, int audio_sec) {
  const int sample_rate = 16000;
  const size_t sample_count = (size_t)sample_rate * (size_t)audio_sec;
  int16_t *samples = calloc(sample_count, sizeof(*samples));
  if (!samples) return -1;

  SttAudioBuffer audio = {
    .samples = samples,
    .len = sample_count,
    .cap = sample_count,
    .sample_rate = sample_rate,
    .channels = 1,
  };
  SttFeatures features;
  OrtValue *encoded = NULL;
  int *ids = NULL;
  size_t id_count = 0;
  int64_t encoded_len = 0;
  long long encoder_ms = 0;
  DecodeStats decode_stats = {0};
  int rc = -1;

  long long start_ms = stt_now_ms();
  long long feature_start_ms = stt_now_ms();
  if (stt_extract_features(&audio, &features) != 0) goto done_free_audio;
  long long feature_ms = stt_now_ms() - feature_start_ms;
  if (encoder_run(&runtime->ort, &features, &encoded, &encoded_len, &encoder_ms) != 0) goto done;
  if (tdt_greedy_decode_onnx(&runtime->ort, encoded, encoded_len, &ids, &id_count, &decode_stats) != 0) goto done;
  rc = 0;

done:
  free(ids);
  if (encoded) runtime->ort.api->ReleaseValue(encoded);
  stt_features_free(&features);
  LOG_DEBUG("infer: warmup_prime audio_sec=%d elapsed_ms=%lld features_ms=%lld encoder_ms=%lld decoder_ms=%lld decoder_calls=%lld output_ids=%zu rc=%d\n",
            audio_sec,
           stt_now_ms() - start_ms,
           rc == 0 ? feature_ms : 0,
           encoder_ms,
           decode_stats.decoder_ms,
           (long long)decode_stats.decoder_calls,
           id_count,
           rc);
done_free_audio:
  free(samples);
  return rc;
}

static int tdt_runtime_prime(TdtRuntime *runtime) {
  static const int durations[] = {1, 2, 4, 8};
  for (size_t i = 0; i < sizeof(durations) / sizeof(durations[0]); ++i) {
    int rc = tdt_runtime_prime_duration(runtime, durations[i]);
    if (rc != 0) return rc;
  }
  return 0;
}

int stt_transcribe_warmup(SttModel *model) {
  long long start_ms = stt_now_ms();
  TdtRuntime *runtime = NULL;
  int rc = tdt_runtime_get(stt_model_dir(model), &runtime);
  if (rc == 0) rc = tdt_runtime_prime(runtime);
  LOG_INFO("infer: warmup elapsed_ms=%lld rc=%d\n", stt_now_ms() - start_ms, rc);
  return rc;
}

int stt_transcribe(SttModel *model, const SttAudioBuffer *audio, char **text_out) {
  static unsigned long long transcribe_seq = 0;
  unsigned long long transcribe_id = ++transcribe_seq;
  double seconds = audio->sample_rate ? (double)audio->len / (double)audio->sample_rate : 0.0;
  LOG_TRACE("infer: start id=%llu audio_sec=%.2f samples=%zu sample_rate=%d channels=%d\n",
            transcribe_id, seconds, audio->len, audio->sample_rate, audio->channels);
  return transcribe_tdt_onnx(model, audio, text_out);
}
