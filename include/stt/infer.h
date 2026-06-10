#ifndef STT_INFER_H
#define STT_INFER_H

#include "stt/audio.h"
#include "stt/model.h"

int stt_transcribe_warmup(SttModel *model);
int stt_transcribe(SttModel *model, const SttAudioBuffer *audio, char **text_out);

#endif
