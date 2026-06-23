#include "stt/cli.h"

#include <assert.h>
#include <string.h>

static void test_defaults(void) {
  char *argv[] = {"stt"};
  SttConfig config;
  assert(stt_parse_args(1, argv, &config) == 0);
  assert(config.model_dir == NULL);
  assert(strcmp(config.infer_provider, "auto") == 0);
  assert(strcmp(config.model_variant, "auto") == 0);
  assert(config.device_id == 0);
  assert(config.threads == 1);
}

static void test_provider_flags(void) {
  char *argv[] = {
    "stt",
    "run",
    "--infer-provider",
    "cuda",
    "--device-id",
    "2",
    "--threads",
    "4",
    "--model-variant",
    "int8",
    "--print",
  };
  SttConfig config;
  assert(stt_parse_args((int)(sizeof(argv) / sizeof(argv[0])), argv, &config) == 0);
  assert(strcmp(config.infer_provider, "cuda") == 0);
  assert(strcmp(config.model_variant, "int8") == 0);
  assert(config.device_id == 2);
  assert(config.threads == 4);
  assert(config.print_only == 1);
}

static void test_invalid_int(void) {
  char *argv[] = {"stt", "--threads", "nope"};
  SttConfig config;
  assert(stt_parse_args((int)(sizeof(argv) / sizeof(argv[0])), argv, &config) != 0);
}

int main(void) {
  test_defaults();
  test_provider_flags();
  test_invalid_int();
  return 0;
}
