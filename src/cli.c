#define _POSIX_C_SOURCE 200809L
#include "stt/cli.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const char *name;
  SttCommand command;
} CommandDef;

static const CommandDef COMMANDS[] = {
  {"run", STT_CMD_RUN},
  {"install-model", STT_CMD_INSTALL_MODEL},
  {"inspect-model", STT_CMD_INSPECT_MODEL},
  {"test-gpu", STT_CMD_TEST_GPU},
};

static void options_defaults(SttOptions *opts) {
  memset(opts, 0, sizeof(*opts));
  opts->command = STT_CMD_NONE;
  opts->hotkey = "Super+V";
  opts->type_delay_ms = 0;
  opts->max_audio_sec = 25;
  opts->pre_roll_ms = 350;
  opts->post_roll_ms = 0;
}

static int parse_int(const char *s, int *out) {
  char *end = NULL;
  long v = strtol(s, &end, 10);
  if (!s[0] || (end && *end)) return -1;
  *out = (int)v;
  return 0;
}

static SttCommand parse_command(const char *name) {
  for (size_t i = 0; i < sizeof(COMMANDS) / sizeof(COMMANDS[0]); ++i) {
    if (strcmp(name, COMMANDS[i].name) == 0) return COMMANDS[i].command;
  }
  return STT_CMD_NONE;
}

int stt_parse_args(int argc, char **argv, SttOptions *opts) {
  options_defaults(opts);
  if (argc < 2) return -1;

  opts->command = parse_command(argv[1]);
  if (opts->command == STT_CMD_NONE) return -1;
  opts->model_dir = opts->command == STT_CMD_RUN ? "~/.models/parakeet-tdt" : "~/.models/parakeet-tdt-0.6b-v3";

  for (int i = 2; i < argc; ++i) {
    if (strcmp(argv[i], "--model-dir") == 0 && i + 1 < argc) {
      opts->model_dir = argv[++i];
    } else if (strcmp(argv[i], "--hotkey") == 0 && i + 1 < argc) {
      opts->hotkey = argv[++i];
    } else if (strcmp(argv[i], "--type-delay-ms") == 0 && i + 1 < argc) {
      if (parse_int(argv[++i], &opts->type_delay_ms) != 0) return -1;
    } else if (strcmp(argv[i], "--max-audio-sec") == 0 && i + 1 < argc) {
      if (parse_int(argv[++i], &opts->max_audio_sec) != 0) return -1;
    } else if (strcmp(argv[i], "--pre-roll-ms") == 0 && i + 1 < argc) {
      if (parse_int(argv[++i], &opts->pre_roll_ms) != 0) return -1;
    } else if (strcmp(argv[i], "--post-roll-ms") == 0 && i + 1 < argc) {
      if (parse_int(argv[++i], &opts->post_roll_ms) != 0) return -1;
    } else if (strcmp(argv[i], "--dry-run") == 0) {
      opts->dry_run = 1;
    } else if (strcmp(argv[i], "--print") == 0) {
      opts->print_only = 1;
    } else {
      return -1;
    }
  }
  return 0;
}

void stt_print_usage(const char *argv0) {
  fprintf(stderr,
          "usage:\n"
          "  %s install-model [--model-dir DIR]\n"
          "  %s inspect-model [--model-dir DIR]\n"
          "  %s test-gpu\n"
          "  %s run [--model-dir DIR] [--hotkey Super+V] [--type-delay-ms N] [--max-audio-sec N] [--pre-roll-ms N] [--post-roll-ms N] [--dry-run|--print]\n",
          argv0, argv0, argv0, argv0);
}
