#ifndef STT_CLI_H
#define STT_CLI_H

#include "stt/config.h"

int stt_parse_args(int argc, char **argv, SttConfig *config);
void stt_print_usage(const char *argv0);

#endif
