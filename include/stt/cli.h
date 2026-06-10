#ifndef STT_CLI_H
#define STT_CLI_H

typedef struct {
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
