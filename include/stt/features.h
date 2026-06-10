#ifndef STT_FEATURES_H
#define STT_FEATURES_H

#include <stddef.h>
#include "stt/audio.h"

typedef struct {
  float *data;
  size_t frames;
  size_t valid_frames;
  size_t bins;
} SttFeatures;

int stt_extract_features(const SttAudioBuffer *audio, SttFeatures *out);
void stt_features_free(SttFeatures *features);

#endif
