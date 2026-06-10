#ifndef STT_MODEL_H
#define STT_MODEL_H

typedef struct SttModel SttModel;

int stt_model_load(SttModel **model_out, const char *model_dir);
void stt_model_free(SttModel *model);
const char *stt_model_dir(const SttModel *model);

#endif
