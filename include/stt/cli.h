#ifndef STT_CLI_H
#define STT_CLI_H

typedef enum {
  STT_CMD_NONE = 0,
  STT_CMD_RUN,
  STT_CMD_INSTALL_MODEL,
  STT_CMD_INSPECT_MODEL,
  STT_CMD_TEST_GPU,
} SttCommand;

typedef struct {
  SttCommand command;
  const char *model_dir;
  const char *hotkey;
  int type_delay_ms;
  int max_audio_sec;
  int pre_roll_ms;
  int post_roll_ms;
  int dry_run;
  int print_only;
} SttOptions;

int stt_parse_args(int argc, char **argv, SttOptions *opts);
void stt_print_usage(const char *argv0);

#endif
