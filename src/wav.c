#define _POSIX_C_SOURCE 200809L
#include "stt/wav.h"
#include "stt/config.h"
#include "stt/log.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int read_u32le(FILE *f, uint32_t *out) {
  unsigned char b[4];
  if (fread(b, 1, 4, f) != 4) return -1;
  *out = (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
  return 0;
}

static int read_u16le(FILE *f, uint16_t *out) {
  unsigned char b[2];
  if (fread(b, 1, 2, f) != 2) return -1;
  *out = (uint16_t)b[0] | ((uint16_t)b[1] << 8);
  return 0;
}

int stt_wav_read(const char *path, SttAudioBuffer *audio_out) {
  FILE *f = fopen(path, "rb");
  if (!f) {
    LOG_ERROR("wav: could not open file=%s\n", path);
    return -1;
  }

  char riff_tag[4], wave_tag[4];
  uint32_t riff_size;
  if (fread(riff_tag, 1, 4, f) != 4 || memcmp(riff_tag, "RIFF", 4) != 0 ||
      read_u32le(f, &riff_size) != 0 ||
      fread(wave_tag, 1, 4, f) != 4 || memcmp(wave_tag, "WAVE", 4) != 0) {
    LOG_ERROR("wav: file=%s is not a RIFF/WAVE file\n", path);
    fclose(f);
    return -1;
  }

  int have_fmt = 0;
  uint16_t audio_format = 0, channels = 0, bits_per_sample = 0;
  uint32_t sample_rate = 0;
  int16_t *samples = NULL;
  size_t sample_frames = 0;

  for (;;) {
    char chunk_id[4];
    uint32_t chunk_size;
    if (fread(chunk_id, 1, 4, f) != 4) break;
    if (read_u32le(f, &chunk_size) != 0) break;
    long chunk_start = ftell(f);
    long chunk_end = chunk_start + (long)chunk_size + (long)(chunk_size & 1);

    if (memcmp(chunk_id, "fmt ", 4) == 0) {
      if (read_u16le(f, &audio_format) != 0 || read_u16le(f, &channels) != 0 ||
          read_u32le(f, &sample_rate) != 0) {
        LOG_ERROR("wav: file=%s has a malformed fmt chunk\n", path);
        fclose(f);
        return -1;
      }
      uint32_t byte_rate;
      uint16_t block_align;
      if (read_u32le(f, &byte_rate) != 0 || read_u16le(f, &block_align) != 0 ||
          read_u16le(f, &bits_per_sample) != 0) {
        LOG_ERROR("wav: file=%s has a malformed fmt chunk\n", path);
        fclose(f);
        return -1;
      }
      have_fmt = 1;
      if (fseek(f, chunk_end, SEEK_SET) != 0) break;
    } else if (memcmp(chunk_id, "data", 4) == 0) {
      if (!have_fmt) {
        LOG_ERROR("wav: file=%s has a data chunk before the fmt chunk\n", path);
        fclose(f);
        return -1;
      }
      if (audio_format != 1 || bits_per_sample != 16) {
        LOG_ERROR("wav: file=%s is not 16-bit PCM (format=%u bits=%u); convert with e.g. "
                   "`sox in.wav -r 16000 -c 1 -b 16 out.wav`\n",
                   path, (unsigned)audio_format, (unsigned)bits_per_sample);
        fclose(f);
        return -1;
      }
      if (sample_rate != STT_SAMPLE_RATE) {
        LOG_ERROR("wav: file=%s sample_rate=%u must be %d Hz; convert with e.g. "
                   "`sox in.wav -r 16000 -c 1 -b 16 out.wav`\n",
                   path, sample_rate, STT_SAMPLE_RATE);
        fclose(f);
        return -1;
      }
      if (channels != 1 && channels != 2) {
        LOG_ERROR("wav: file=%s has channels=%u; only mono or stereo is supported\n", path, (unsigned)channels);
        fclose(f);
        return -1;
      }
      size_t bytes_per_frame = (size_t)channels * 2;
      if (chunk_size % bytes_per_frame != 0) {
        LOG_ERROR("wav: file=%s data chunk is not a whole number of sample frames\n", path);
        fclose(f);
        return -1;
      }
      sample_frames = chunk_size / bytes_per_frame;
      int16_t *raw = malloc(chunk_size);
      if (!raw || fread(raw, 1, chunk_size, f) != chunk_size) {
        LOG_ERROR("wav: file=%s data chunk is shorter than declared\n", path);
        free(raw);
        fclose(f);
        return -1;
      }
      if (channels == 1) {
        samples = raw;
      } else {
        samples = malloc(sample_frames * sizeof(int16_t));
        if (!samples) {
          free(raw);
          fclose(f);
          return -1;
        }
        for (size_t i = 0; i < sample_frames; ++i) {
          int32_t l = raw[i * 2];
          int32_t r = raw[i * 2 + 1];
          samples[i] = (int16_t)((l + r) / 2);
        }
        free(raw);
      }
      break;
    } else {
      if (fseek(f, chunk_end, SEEK_SET) != 0) break;
    }
  }
  fclose(f);

  if (!samples) {
    LOG_ERROR("wav: file=%s has no usable PCM data chunk\n", path);
    return -1;
  }

  memset(audio_out, 0, sizeof(*audio_out));
  audio_out->samples = samples;
  audio_out->len = sample_frames;
  audio_out->cap = sample_frames;
  audio_out->sample_rate = STT_SAMPLE_RATE;
  audio_out->channels = 1;
  return 0;
}
