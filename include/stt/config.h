#ifndef STT_CONFIG_H
#define STT_CONFIG_H

#define STT_SAMPLE_RATE 16000

typedef struct {
  const char *model_dir;
  const char *log_path;
  int type_delay_ms;
  int max_audio_sec;
  int pre_roll_ms;
  int post_roll_ms;
  int dry_run;
  int print_only;
} SttConfig;

#endif
