#ifndef STT_MODEL_H
#define STT_MODEL_H

typedef struct SttModel SttModel;

typedef enum {
  STT_MODEL_UNKNOWN = 0,
  STT_MODEL_TDT_ONNX,
  STT_MODEL_V3_SAFETENSORS,
} SttModelKind;

int stt_model_load(SttModel **model_out, const char *model_dir);
void stt_model_free(SttModel *model);
int stt_model_inspect(const char *model_dir);
SttModelKind stt_model_kind(const SttModel *model);
const char *stt_model_dir(const SttModel *model);

#endif
