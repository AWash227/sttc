#define _POSIX_C_SOURCE 200809L
#include "stt/wav.h"
#include "stt/config.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#define stt_unlink _unlink
#else
#include <unistd.h>
#define stt_unlink unlink
#endif

static void write_u32le(FILE *f, uint32_t v) {
  unsigned char b[4] = {(unsigned char)v, (unsigned char)(v >> 8), (unsigned char)(v >> 16), (unsigned char)(v >> 24)};
  fwrite(b, 1, 4, f);
}

static void write_u16le(FILE *f, uint16_t v) {
  unsigned char b[2] = {(unsigned char)v, (unsigned char)(v >> 8)};
  fwrite(b, 1, 2, f);
}

/* Writes a minimal canonical PCM WAV file with the given sample rate,
 * channel count, and interleaved 16-bit samples. */
static void write_wav(const char *path, uint32_t sample_rate, uint16_t channels, const int16_t *samples, size_t frame_count) {
  FILE *f = fopen(path, "wb");
  assert(f);
  uint32_t data_bytes = (uint32_t)(frame_count * channels * sizeof(int16_t));
  uint16_t block_align = (uint16_t)(channels * sizeof(int16_t));
  uint32_t byte_rate = sample_rate * block_align;

  fwrite("RIFF", 1, 4, f);
  write_u32le(f, 36 + data_bytes);
  fwrite("WAVE", 1, 4, f);

  fwrite("fmt ", 1, 4, f);
  write_u32le(f, 16);
  write_u16le(f, 1); /* PCM */
  write_u16le(f, channels);
  write_u32le(f, sample_rate);
  write_u32le(f, byte_rate);
  write_u16le(f, block_align);
  write_u16le(f, 16);

  fwrite("data", 1, 4, f);
  write_u32le(f, data_bytes);
  fwrite(samples, 1, data_bytes, f);
  fclose(f);
}

static char *temp_path(const char *name) {
  char buf[512];
#ifdef _WIN32
  const char *base = getenv("TEMP");
  if (!base || !*base) base = ".";
  snprintf(buf, sizeof(buf), "%s\\%s-%lu", base, name, (unsigned long)GetCurrentProcessId());
#else
  snprintf(buf, sizeof(buf), "/tmp/%s-%d", name, (int)getpid());
#endif
  return strdup(buf);
}

static void test_reads_mono_pcm(void) {
  int16_t samples[] = {100, -200, 300, -400};
  char *path = temp_path("stt-wav-mono");
  write_wav(path, STT_SAMPLE_RATE, 1, samples, 4);

  SttAudioBuffer audio;
  assert(stt_wav_read(path, &audio) == 0);
  assert(audio.sample_rate == STT_SAMPLE_RATE);
  assert(audio.channels == 1);
  assert(audio.len == 4);
  for (int i = 0; i < 4; ++i) assert(audio.samples[i] == samples[i]);

  stt_audio_buffer_free(&audio);
  stt_unlink(path);
  free(path);
}

static void test_downmixes_stereo(void) {
  /* Interleaved L,R pairs: (100,300)->200 and (-100,-300)->-200 */
  int16_t samples[] = {100, 300, -100, -300};
  char *path = temp_path("stt-wav-stereo");
  write_wav(path, STT_SAMPLE_RATE, 2, samples, 2);

  SttAudioBuffer audio;
  assert(stt_wav_read(path, &audio) == 0);
  assert(audio.channels == 1);
  assert(audio.len == 2);
  assert(audio.samples[0] == 200);
  assert(audio.samples[1] == -200);

  stt_audio_buffer_free(&audio);
  stt_unlink(path);
  free(path);
}

static void test_rejects_wrong_sample_rate(void) {
  int16_t samples[] = {1, 2, 3, 4};
  char *path = temp_path("stt-wav-badrate");
  write_wav(path, 44100, 1, samples, 4);

  SttAudioBuffer audio;
  assert(stt_wav_read(path, &audio) != 0);

  stt_unlink(path);
  free(path);
}

static void test_rejects_non_riff_file(void) {
  char *path = temp_path("stt-wav-notriff");
  FILE *f = fopen(path, "wb");
  assert(f);
  fwrite("not a wav file", 1, 14, f);
  fclose(f);

  SttAudioBuffer audio;
  assert(stt_wav_read(path, &audio) != 0);

  stt_unlink(path);
  free(path);
}

static void test_rejects_missing_file(void) {
  SttAudioBuffer audio;
  assert(stt_wav_read("/nonexistent/stt-wav-missing.wav", &audio) != 0);
}

int main(void) {
  test_reads_mono_pcm();
  test_downmixes_stereo();
  test_rejects_wrong_sample_rate();
  test_rejects_non_riff_file();
  test_rejects_missing_file();
  return 0;
}
