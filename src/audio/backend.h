#ifndef STT_AUDIO_BACKEND_INTERNAL_H
#define STT_AUDIO_BACKEND_INTERNAL_H

#include "stt/config.h"

#include <stddef.h>
#include <stdint.h>

typedef struct SttAudioSource SttAudioSource;

const char *stt_audio_backend_name(void);
int stt_audio_source_open(SttAudioSource **out, const SttConfig *config, const char *stream_name);
void stt_audio_source_close(SttAudioSource *source);
int stt_audio_source_read(SttAudioSource *source, int16_t *samples, size_t sample_count);

#endif
