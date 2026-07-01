#define _POSIX_C_SOURCE 200809L
#include "stt/cli.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void config_defaults(SttConfig *config) {
  memset(config, 0, sizeof(*config));
  config->infer_provider = "auto";
  config->model_variant = "auto";
  config->type_delay_ms = 0;
  config->max_audio_sec = 25;
  config->pre_roll_ms = 350;
  config->post_roll_ms = 0;
  config->device_id = 0;
  config->threads = 1;
}

static int parse_int(const char *s, int *out) {
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (!s[0] || (end && *end)) return -1;
  *out = (int)v;
  return 0;
}

int stt_parse_args(int argc, char **argv, SttConfig *config) {
  config_defaults(config);
  int arg_start = 1;
  if (argc >= 2 && strcmp(argv[1], "run") == 0) arg_start = 2;

  for (int i = arg_start; i < argc; ++i) {
    if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
      config->model_dir = argv[++i];
    } else if (strcmp(argv[i], "--log") == 0 && i + 1 < argc) {
      config->log_path = argv[++i];
    } else if (strcmp(argv[i], "--type-delay-ms") == 0 && i + 1 < argc) {
      if (parse_int(argv[++i], &config->type_delay_ms) != 0) return -1;
    } else if (strcmp(argv[i], "--max-audio-sec") == 0 && i + 1 < argc) {
      if (parse_int(argv[++i], &config->max_audio_sec) != 0) return -1;
    } else if (strcmp(argv[i], "--pre-roll-ms") == 0 && i + 1 < argc) {
      if (parse_int(argv[++i], &config->pre_roll_ms) != 0) return -1;
    } else if (strcmp(argv[i], "--post-roll-ms") == 0 && i + 1 < argc) {
      if (parse_int(argv[++i], &config->post_roll_ms) != 0) return -1;
    } else if (strcmp(argv[i], "--infer-provider") == 0 && i + 1 < argc) {
      config->infer_provider = argv[++i];
    } else if (strcmp(argv[i], "--device-id") == 0 && i + 1 < argc) {
      if (parse_int(argv[++i], &config->device_id) != 0) return -1;
    } else if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
      if (parse_int(argv[++i], &config->threads) != 0) return -1;
    } else if (strcmp(argv[i], "--model-variant") == 0 && i + 1 < argc) {
      config->model_variant = argv[++i];
    } else if (strcmp(argv[i], "--dry-run") == 0) {
      config->dry_run = 1;
    } else if (strcmp(argv[i], "--print") == 0) {
      config->print_only = 1;
    } else if (argv[i][0] != '-' && !config->input_file) {
      config->input_file = argv[i];
    } else {
      return -1;
    }
  }
  return 0;
}

void stt_print_usage(const char *argv0) {
  fprintf(stderr,
          "usage:\n"
          "  %s [run] [--model-dir DIR] [--log FILE] [--type-delay-ms N] [--max-audio-sec N] [--pre-roll-ms N] [--post-roll-ms N]\n"
          "     [--infer-provider auto|cpu|cuda|directml|coreml|openvino|migraphx|xnnpack] [--device-id N] [--threads N]\n"
          "     [--model-variant auto|fp32|int8] [--dry-run|--print]\n"
          "  %s FILE.wav [options...]\n"
          "\n"
          "  Passing a bare file path transcribes that 16-bit PCM WAV file (mono or stereo, %d Hz) and prints the\n"
          "  result to stdout, instead of listening for the hotkey.\n",
          argv0, argv0, STT_SAMPLE_RATE);
}
