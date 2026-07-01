#ifndef STT_CONFIG_H
#define STT_CONFIG_H

#define STT_SAMPLE_RATE 16000

typedef struct {
  const char *model_dir;
  const char *log_path;
  const char *input_file;
  const char *infer_provider;
  const char *model_variant;
  int type_delay_ms;
  int max_audio_sec;
  int pre_roll_ms;
  int post_roll_ms;
  int device_id;
  int threads;
  int dry_run;
  int print_only;
} SttConfig;

#endif
