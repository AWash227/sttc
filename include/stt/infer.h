#ifndef STT_INFER_H
#define STT_INFER_H

#include "stt/audio.h"
#include "stt/config.h"
#include "stt/model.h"

int stt_transcribe_warmup(SttModel *model, const SttConfig *config);
int stt_transcribe(SttModel *model, const SttAudioBuffer *audio, const SttConfig *config, char **text_out);

#endif
