#define _POSIX_C_SOURCE 200809L
#include "stt/infer.h"
#include "stt/features.h"
#include "stt/fs.h"
#include "stt/log.h"

#include <onnxruntime_c_api.h>
#include <errno.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#endif

#define TDT_FEATURE_BINS 128
#define TDT_ENCODER_DIM 1024
#define TDT_STATE_LAYERS 2
#define TDT_STATE_DIM 640
#define TDT_DURATION_COUNT 5
#define TDT_MAX_TOKENS_PER_STEP 10

typedef struct {
  char **tokens;
  size_t count;
  int blank_id;
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
  char *provider;
  TdtVocab vocab;
  OrtTdt ort;
  int ready;
} TdtRuntime;

static TdtRuntime g_tdt_runtime;

static double elapsed_rtf(long long elapsed_ms, double audio_sec) {
  return audio_sec > 0.0 ? (double)elapsed_ms / (audio_sec * 1000.0) : 0.0;
}

typedef enum {
  STT_INFER_PROVIDER_AUTO = 0,
  STT_INFER_PROVIDER_CPU,
  STT_INFER_PROVIDER_CUDA,
  STT_INFER_PROVIDER_DIRECTML,
  STT_INFER_PROVIDER_COREML,
  STT_INFER_PROVIDER_OPENVINO,
  STT_INFER_PROVIDER_MIGRAPHX,
  STT_INFER_PROVIDER_XNNPACK,
  STT_INFER_PROVIDER_UNKNOWN,
} SttInferProvider;

static int streq(const char *a, const char *b) {
  return a && b && strcmp(a, b) == 0;
}

static SttInferProvider parse_provider(const char *provider) {
  if (!provider || !*provider || streq(provider, "auto")) return STT_INFER_PROVIDER_AUTO;
  if (streq(provider, "cpu")) return STT_INFER_PROVIDER_CPU;
  if (streq(provider, "cuda")) return STT_INFER_PROVIDER_CUDA;
  if (streq(provider, "directml")) return STT_INFER_PROVIDER_DIRECTML;
  if (streq(provider, "coreml")) return STT_INFER_PROVIDER_COREML;
  if (streq(provider, "openvino")) return STT_INFER_PROVIDER_OPENVINO;
  if (streq(provider, "migraphx")) return STT_INFER_PROVIDER_MIGRAPHX;
  if (streq(provider, "xnnpack")) return STT_INFER_PROVIDER_XNNPACK;
  return STT_INFER_PROVIDER_UNKNOWN;
}

static const char *provider_name(SttInferProvider provider) {
  switch (provider) {
    case STT_INFER_PROVIDER_AUTO: return "auto";
    case STT_INFER_PROVIDER_CPU: return "cpu";
    case STT_INFER_PROVIDER_CUDA: return "cuda";
    case STT_INFER_PROVIDER_DIRECTML: return "directml";
    case STT_INFER_PROVIDER_COREML: return "coreml";
    case STT_INFER_PROVIDER_OPENVINO: return "openvino";
    case STT_INFER_PROVIDER_MIGRAPHX: return "migraphx";
    case STT_INFER_PROVIDER_XNNPACK: return "xnnpack";
    case STT_INFER_PROVIDER_UNKNOWN: return "unknown";
  }
  return "unknown";
}

static const char *requested_provider_name(const SttConfig *config) {
  return config && config->infer_provider ? config->infer_provider : "auto";
}

static const char *requested_tdt_variant(const SttConfig *config, SttInferProvider provider) {
  const char *variant = config ? config->model_variant : NULL;
  if (!variant || !*variant) variant = getenv("STT_TDT_VARIANT");
  if (!variant || !*variant || strcmp(variant, "auto") == 0) {
    if (provider == STT_INFER_PROVIDER_CPU || provider == STT_INFER_PROVIDER_CUDA) return "fp32";
    return "fp32";
  }
  if (strcmp(variant, "fp32") == 0 || strcmp(variant, "int8") == 0) return variant;
  LOG_WARN("infer: unknown model variant=%s using=fp32\n", variant);
  return "fp32";
}

static int file_exists_in_dir(const char *dir, const char *name) {
  char *path = stt_path_join(dir, name);
  if (!path) return 0;
  int exists = stt_file_exists(path);
  free(path);
  return exists;
}

static void tdt_variant_files(const char *model_dir,
                              const SttConfig *config,
                              SttInferProvider provider,
                              const char **variant_out,
                              const char **encoder_file_out,
                              const char **decoder_file_out) {
  const char *variant = requested_tdt_variant(config, provider);
  const char *encoder_file = "encoder-model.onnx";
  const char *decoder_file = "decoder_joint-model.onnx";

  if (strcmp(variant, "int8") == 0) {
    if (provider != STT_INFER_PROVIDER_CPU && provider != STT_INFER_PROVIDER_CUDA) {
      LOG_WARN("infer: tdt_variant=int8 is not verified for provider=%s; falling back to fp32\n", provider_name(provider));
      variant = "fp32";
    } else if (file_exists_in_dir(model_dir, "encoder-model.int8.onnx") &&
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

static int provider_compiled(SttInferProvider provider) {
  switch (provider) {
    case STT_INFER_PROVIDER_CPU:
      return 1;
    case STT_INFER_PROVIDER_CUDA:
      return STT_ENABLE_CUDA;
    case STT_INFER_PROVIDER_DIRECTML:
      return STT_ENABLE_DIRECTML;
    case STT_INFER_PROVIDER_COREML:
      return STT_ENABLE_COREML;
    case STT_INFER_PROVIDER_OPENVINO:
      return STT_ENABLE_OPENVINO;
    case STT_INFER_PROVIDER_MIGRAPHX:
      return STT_ENABLE_MIGRAPHX;
    case STT_INFER_PROVIDER_XNNPACK:
      return STT_ENABLE_XNNPACK;
    case STT_INFER_PROVIDER_AUTO:
    case STT_INFER_PROVIDER_UNKNOWN:
      return 0;
  }
  return 0;
}

static void log_available_providers(OrtTdt *ort) {
  char **providers = NULL;
  int len = 0;
  OrtStatus *status = ort->api->GetAvailableProviders(&providers, &len);
  if (status) {
    LOG_DEBUG("infer: available_providers error=\"%s\"\n", ort->api->GetErrorMessage(status));
    ort->api->ReleaseStatus(status);
    return;
  }
  for (int i = 0; i < len; ++i) {
    LOG_DEBUG("infer: available_provider=%s\n", providers[i]);
  }
  OrtStatus *release_status = ort->api->ReleaseAvailableProviders(providers, len);
  if (release_status) ort->api->ReleaseStatus(release_status);
}

static int provider_status(OrtTdt *ort,
                           OrtStatus *status,
                           SttInferProvider provider,
                           int explicit_request,
                           const char *provider_detail,
                           const char **selected_provider_out) {
  if (!status) {
    *selected_provider_out = provider_name(provider);
    LOG_DEBUG("infer: provider=%s configured detail=%s\n", provider_name(provider), provider_detail ? provider_detail : "");
    return 1;
  }

  const char *message = ort->api->GetErrorMessage(status);
  if (explicit_request) {
    LOG_ERROR("infer: requested provider=%s unavailable: %s\n", provider_name(provider), message);
    ort->api->ReleaseStatus(status);
    return -1;
  }
  LOG_DEBUG("infer: provider=%s unavailable reason=\"%s\"\n", provider_name(provider), message);
  ort->api->ReleaseStatus(status);
  return 0;
}

static int append_generic_provider(OrtTdt *ort,
                                   SttInferProvider provider,
                                   const char *ort_provider_name,
                                   const char *option_key,
                                   const char *option_value,
                                   int explicit_request,
                                   const char **selected_provider_out) {
  if (!provider_compiled(provider)) {
    if (explicit_request) {
      LOG_ERROR("infer: requested provider=%s is not enabled in this build\n", provider_name(provider));
      return -1;
    }
    LOG_DEBUG("infer: provider=%s skipped reason=not_enabled\n", provider_name(provider));
    return 0;
  }
  const char *keys[1] = {option_key};
  const char *values[1] = {option_value};
  size_t count = option_key && option_value ? 1 : 0;
  OrtStatus *status = ort->api->SessionOptionsAppendExecutionProvider(ort->options, ort_provider_name, keys, values, count);
  return provider_status(ort, status, provider, explicit_request, ort_provider_name, selected_provider_out);
}

static int append_provider(OrtTdt *ort,
                           SttInferProvider provider,
                           const SttConfig *config,
                           int explicit_request,
                           const char **selected_provider_out) {
  if (provider == STT_INFER_PROVIDER_CPU) {
    *selected_provider_out = "cpu";
    LOG_DEBUG("infer: provider=cpu configured detail=default\n");
    return 1;
  }
  if (!provider_compiled(provider)) {
    if (explicit_request) {
      LOG_ERROR("infer: requested provider=%s is not enabled in this build\n", provider_name(provider));
      return -1;
    }
    LOG_DEBUG("infer: provider=%s skipped reason=not_enabled\n", provider_name(provider));
    return 0;
  }

  int device_id = config ? config->device_id : 0;
  switch (provider) {
    case STT_INFER_PROVIDER_CUDA: {
      const char *algo_name = NULL;
      OrtCudnnConvAlgoSearch algo_search = cudnn_algo_search_from_env(&algo_name);
      OrtCUDAProviderOptions options = {
        .device_id = device_id,
        .cudnn_conv_algo_search = algo_search,
        .gpu_mem_limit = SIZE_MAX,
        .arena_extend_strategy = 1,
        .do_copy_in_default_stream = 1,
      };
      OrtStatus *status = ort->api->SessionOptionsAppendExecutionProvider_CUDA(ort->options, &options);
      return provider_status(ort, status, provider, explicit_request, algo_name, selected_provider_out);
    }
    case STT_INFER_PROVIDER_MIGRAPHX: {
      OrtMIGraphXProviderOptions options;
      memset(&options, 0, sizeof(options));
      options.device_id = device_id;
      options.migraphx_mem_limit = SIZE_MAX;
      OrtStatus *status = ort->api->SessionOptionsAppendExecutionProvider_MIGraphX(ort->options, &options);
      return provider_status(ort, status, provider, explicit_request, "MIGraphX", selected_provider_out);
    }
    case STT_INFER_PROVIDER_OPENVINO:
      return append_generic_provider(ort, provider, "OpenVINO", NULL, NULL, explicit_request, selected_provider_out);
    case STT_INFER_PROVIDER_DIRECTML:
      return append_generic_provider(ort, provider, "DML", NULL, NULL, explicit_request, selected_provider_out);
    case STT_INFER_PROVIDER_COREML:
      return append_generic_provider(ort, provider, "CoreML", NULL, NULL, explicit_request, selected_provider_out);
    case STT_INFER_PROVIDER_XNNPACK: {
      char threads[16];
      snprintf(threads, sizeof(threads), "%d", config && config->threads > 0 ? config->threads : 1);
      return append_generic_provider(ort, provider, "XNNPACK", "intra_op_num_threads", threads, explicit_request, selected_provider_out);
    }
    case STT_INFER_PROVIDER_AUTO:
    case STT_INFER_PROVIDER_CPU:
    case STT_INFER_PROVIDER_UNKNOWN:
      break;
  }
  return 0;
}

static int configure_providers(OrtTdt *ort, const SttConfig *config, const char **selected_provider_out) {
  SttInferProvider requested = parse_provider(requested_provider_name(config));
  if (requested == STT_INFER_PROVIDER_UNKNOWN) {
    LOG_ERROR("infer: unknown provider=%s\n", requested_provider_name(config));
    return -1;
  }

  log_available_providers(ort);
  if (requested != STT_INFER_PROVIDER_AUTO) {
    return append_provider(ort, requested, config, 1, selected_provider_out) < 0 ? -1 : 0;
  }

  static const SttInferProvider auto_order[] = {
    STT_INFER_PROVIDER_CUDA,
    STT_INFER_PROVIDER_DIRECTML,
    STT_INFER_PROVIDER_COREML,
    STT_INFER_PROVIDER_OPENVINO,
    STT_INFER_PROVIDER_MIGRAPHX,
    STT_INFER_PROVIDER_XNNPACK,
  };
  for (size_t i = 0; i < sizeof(auto_order) / sizeof(auto_order[0]); ++i) {
    int rc = append_provider(ort, auto_order[i], config, 0, selected_provider_out);
    if (rc < 0) return -1;
    if (rc > 0) return 0;
  }
  *selected_provider_out = "cpu";
  LOG_DEBUG("infer: provider=cpu configured detail=auto_fallback\n");
  return 0;
}

static void tdt_vocab_free(TdtVocab *vocab) {
  if (!vocab) return;
  for (size_t i = 0; i < vocab->count; ++i) free(vocab->tokens[i]);
  free(vocab->tokens);
  memset(vocab, 0, sizeof(*vocab));
}

static int tdt_vocab_load(TdtVocab *vocab, const char *path) {
  memset(vocab, 0, sizeof(*vocab));
  vocab->blank_id = -1;
  size_t len = 0;
  char *text = stt_read_file(path, &len);
  if (!text) {
    LOG_ERROR("failed to read vocab: %s: %s\n", path, strerror(errno));
    return -1;
  }

  size_t cap = 1024;
  vocab->tokens = calloc(cap, sizeof(*vocab->tokens));
  if (!vocab->tokens) {
    free(text);
    return -1;
  }

  char *save = NULL;
  for (char *line = strtok_r(text, "\n", &save); line; line = strtok_r(NULL, "\n", &save)) {
    char *sep = strrchr(line, ' ');
    if (!sep) continue;
    *sep++ = '\0';
    char *end = NULL;
    long id = strtol(sep, &end, 10);
    if (*line && id >= 0 && end && (*end == '\0' || *end == '\r')) {
      size_t token_id = (size_t)id;
      if (token_id >= cap) {
        size_t next_cap = cap;
        while (token_id >= next_cap) next_cap *= 2;
        char **next = realloc(vocab->tokens, next_cap * sizeof(*next));
        if (!next) {
          free(text);
          tdt_vocab_free(vocab);
          return -1;
        }
        memset(next + cap, 0, (next_cap - cap) * sizeof(*next));
        vocab->tokens = next;
        cap = next_cap;
      }
      vocab->tokens[id] = strdup(line);
      if (!vocab->tokens[id]) {
        free(text);
        tdt_vocab_free(vocab);
        return -1;
      }
      if (strcmp(line, "<blk>") == 0) vocab->blank_id = (int)id;
      if (token_id + 1 > vocab->count) vocab->count = token_id + 1;
    }
  }

  free(text);
  if (vocab->blank_id < 0 || (size_t)vocab->blank_id >= vocab->count || !vocab->tokens[vocab->blank_id]) {
    tdt_vocab_free(vocab);
    return -1;
  }
  return 0;
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
  free(runtime->provider);
  memset(runtime, 0, sizeof(*runtime));
}

#ifdef _WIN32
static wchar_t *utf8_to_wide(const char *path) {
  int needed = MultiByteToWideChar(CP_UTF8, 0, path, -1, NULL, 0);
  if (needed <= 0) return NULL;
  wchar_t *wide = malloc((size_t)needed * sizeof(*wide));
  if (!wide) return NULL;
  if (MultiByteToWideChar(CP_UTF8, 0, path, -1, wide, needed) <= 0) {
    free(wide);
    return NULL;
  }
  return wide;
}
#endif

static int ort_tdt_init(OrtTdt *ort,
                        const SttConfig *config,
                        const char *model_dir,
                        const char *variant,
                        const char *encoder_file,
                        const char *decoder_file,
                        const char **selected_provider_out) {
  long long start_ms = stt_now_ms();
  memset(ort, 0, sizeof(*ort));
  ort->api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  char *encoder_path = stt_path_join(model_dir, encoder_file);
  char *decoder_path = stt_path_join(model_dir, decoder_file);
#ifdef _WIN32
  wchar_t *encoder_path_w = NULL;
  wchar_t *decoder_path_w = NULL;
#endif
  if (!encoder_path || !decoder_path) goto fail;
#ifdef _WIN32
  encoder_path_w = utf8_to_wide(encoder_path);
  decoder_path_w = utf8_to_wide(decoder_path);
  if (!encoder_path_w || !decoder_path_w) goto fail;
#endif
  LOG_DEBUG("infer: tdt_variant=%s encoder_file=%s decoder_file=%s\n", variant, encoder_file, decoder_file);

  if (ort_check(ort, ort->api->CreateEnv(ORT_LOGGING_LEVEL_ERROR, "stt", &ort->env), "CreateEnv") != 0) goto fail;
  if (ort_check(ort, ort->api->CreateSessionOptions(&ort->options), "CreateSessionOptions") != 0) goto fail;
  int threads = config && config->threads > 0 ? config->threads : 1;
  ort_check(ort, ort->api->SetIntraOpNumThreads(ort->options, threads), "SetIntraOpNumThreads");
  ort_check(ort, ort->api->SetSessionGraphOptimizationLevel(ort->options, ORT_ENABLE_ALL), "SetSessionGraphOptimizationLevel");
  if (configure_providers(ort, config, selected_provider_out) != 0) goto fail;
#ifdef _WIN32
  if (ort_check(ort, ort->api->CreateSession(ort->env, encoder_path_w, ort->options, &ort->encoder), "CreateSession encoder") != 0) goto fail;
  if (ort_check(ort, ort->api->CreateSession(ort->env, decoder_path_w, ort->options, &ort->decoder), "CreateSession decoder") != 0) goto fail;
#else
  if (ort_check(ort, ort->api->CreateSession(ort->env, encoder_path, ort->options, &ort->encoder), "CreateSession encoder") != 0) goto fail;
  if (ort_check(ort, ort->api->CreateSession(ort->env, decoder_path, ort->options, &ort->decoder), "CreateSession decoder") != 0) goto fail;
#endif
  if (ort_check(ort, ort->api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &ort->memory), "CreateCpuMemoryInfo") != 0) goto fail;
  LOG_DEBUG("infer: ort_init done elapsed_ms=%lld provider=%s threads=%d device_id=%d\n",
            stt_now_ms() - start_ms, *selected_provider_out, threads, config ? config->device_id : 0);

  free(encoder_path);
  free(decoder_path);
#ifdef _WIN32
  free(encoder_path_w);
  free(decoder_path_w);
#endif
  return 0;

fail:
  free(encoder_path);
  free(decoder_path);
#ifdef _WIN32
  free(encoder_path_w);
  free(decoder_path_w);
#endif
  ort_tdt_free(ort);
  return -1;
}

static int tdt_runtime_get(const char *model_dir, const SttConfig *config, TdtRuntime **runtime_out) {
  long long start_ms = stt_now_ms();
  SttInferProvider requested_provider = parse_provider(requested_provider_name(config));
  if (requested_provider == STT_INFER_PROVIDER_UNKNOWN) {
    LOG_ERROR("infer: unknown provider=%s\n", requested_provider_name(config));
    return -1;
  }
  const char *variant = NULL;
  const char *encoder_file = NULL;
  const char *decoder_file = NULL;
  tdt_variant_files(model_dir, config, requested_provider, &variant, &encoder_file, &decoder_file);

  if (g_tdt_runtime.ready &&
      strcmp(g_tdt_runtime.model_dir, model_dir) == 0 &&
      strcmp(g_tdt_runtime.variant, variant) == 0 &&
      strcmp(g_tdt_runtime.provider, requested_provider_name(config)) == 0) {
    LOG_TRACE("infer: runtime_cache hit=1 elapsed_ms=0\n");
    *runtime_out = &g_tdt_runtime;
    return 0;
  }

  LOG_DEBUG("infer: runtime_cache hit=0 loading model_dir=%s variant=%s provider_request=%s\n",
            model_dir, variant, requested_provider_name(config));
  tdt_runtime_free(&g_tdt_runtime);
  char *vocab_path = stt_path_join(model_dir, "vocab.txt");
  if (!vocab_path) return -1;
  g_tdt_runtime.model_dir = strdup(model_dir);
  g_tdt_runtime.variant = strdup(variant);
  g_tdt_runtime.provider = strdup(requested_provider_name(config));
  if (!g_tdt_runtime.model_dir || !g_tdt_runtime.variant || !g_tdt_runtime.provider) {
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
  const char *selected_provider = "cpu";
  if (ort_tdt_init(&g_tdt_runtime.ort, config, model_dir, variant, encoder_file, decoder_file, &selected_provider) != 0) {
    tdt_runtime_free(&g_tdt_runtime);
    return -1;
  }
  g_tdt_runtime.ready = 1;
  LOG_DEBUG("infer: runtime_cache loaded elapsed_ms=%lld tokens=%zu variant=%s provider_request=%s provider_selected=%s\n",
            stt_now_ms() - start_ms,
            g_tdt_runtime.vocab.count,
            g_tdt_runtime.variant,
            g_tdt_runtime.provider,
            selected_provider);
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
                        const TdtVocab *vocab,
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
  int token = argmax(logits, vocab->count);
  int duration = argmax(logits + vocab->count, TDT_DURATION_COUNT);
  if (token != vocab->blank_id) {
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

static int tdt_greedy_decode_onnx(OrtTdt *ort, const TdtVocab *vocab, OrtValue *encoded_value, int64_t encoded_len, int **ids_out, size_t *id_count_out, DecodeStats *stats) {
  float *encoder = NULL;
  if (get_tensor_data(ort, encoded_value, &encoder) != 0) return -1;

  float state1[TDT_STATE_LAYERS * TDT_STATE_DIM] = {0};
  float state2[TDT_STATE_LAYERS * TDT_STATE_DIM] = {0};
  float next_state1[TDT_STATE_LAYERS * TDT_STATE_DIM];
  float next_state2[TDT_STATE_LAYERS * TDT_STATE_DIM];
  int *ids = NULL;
  size_t id_count = 0;
  size_t id_cap = 0;
  int32_t target = vocab->blank_id;
  int emitted_tokens = 0;
  memset(stats, 0, sizeof(*stats));

  for (int64_t t = 0; t < encoded_len;) {
    int token = vocab->blank_id;
    int duration = 0;
    long long step_start_ms = stt_now_ms();
    if (decoder_step(ort, encoder, encoded_len, t, vocab, target, state1, state2, next_state1, next_state2, &token, &duration) != 0) goto fail;
    stats->decoder_ms += stt_now_ms() - step_start_ms;
    stats->decoder_calls++;

    if (token != vocab->blank_id) {
      if (push_id(&ids, &id_count, &id_cap, token) != 0) goto fail;
      stats->emitted_tokens++;
      memcpy(state1, next_state1, sizeof(state1));
      memcpy(state2, next_state2, sizeof(state2));
      target = token;
      emitted_tokens++;
    }

    if (duration > 0) {
      t += duration;
      stats->advanced_frames += duration;
      emitted_tokens = 0;
    } else if (token == vocab->blank_id || emitted_tokens == TDT_MAX_TOKENS_PER_STEP) {
      t++;
      stats->advanced_frames++;
      if (token == vocab->blank_id) stats->blank_steps++;
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

static int transcribe_tdt_onnx(SttModel *model, const SttAudioBuffer *audio, const SttConfig *config, char **text_out) {
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
  if (tdt_runtime_get(stt_model_dir(model), config, &runtime) != 0) goto done;
  runtime_ms = stt_now_ms() - runtime_start_ms;
  if (encoder_run(&runtime->ort, &features, &encoded, &encoded_len, &encoder_ms) != 0) goto done;
  LOG_TRACE("infer: encoder elapsed_ms=%lld encoded_frames=%lld\n", encoder_ms, (long long)encoded_len);
  if (tdt_greedy_decode_onnx(&runtime->ort, &runtime->vocab, encoded, encoded_len, &ids, &id_count, &decode_stats) != 0) goto done;
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
  LOG_DEBUG("perf: infer total_ms=%lld rtf=%.3f longest=%s longest_ms=%lld features_ms=%lld runtime_ms=%lld encoder_ms=%lld decoder_ms=%lld text_ms=%lld overhead_ms=%lld rc=%d\n",
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

int stt_transcribe_warmup(SttModel *model, const SttConfig *config) {
  long long start_ms = stt_now_ms();
  TdtRuntime *runtime = NULL;
  int rc = tdt_runtime_get(stt_model_dir(model), config, &runtime);
  LOG_DEBUG("infer: warmup elapsed_ms=%lld rc=%d\n", stt_now_ms() - start_ms, rc);
  return rc;
}

int stt_transcribe(SttModel *model, const SttAudioBuffer *audio, const SttConfig *config, char **text_out) {
  static unsigned long long transcribe_seq = 0;
  unsigned long long transcribe_id = ++transcribe_seq;
  double seconds = audio->sample_rate ? (double)audio->len / (double)audio->sample_rate : 0.0;
  LOG_TRACE("infer: start id=%llu audio_sec=%.2f samples=%zu sample_rate=%d channels=%d\n",
            transcribe_id, seconds, audio->len, audio->sample_rate, audio->channels);
  return transcribe_tdt_onnx(model, audio, config, text_out);
}
