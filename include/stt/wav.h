#ifndef STT_WAV_H
#define STT_WAV_H

#include "stt/audio.h"

/* Reads a PCM WAV file into audio_out. Requires 16-bit PCM at STT_SAMPLE_RATE;
 * mono is used as-is, stereo is downmixed to mono. On failure, logs the
 * reason and returns -1. */
int stt_wav_read(const char *path, SttAudioBuffer *audio_out);

#endif
