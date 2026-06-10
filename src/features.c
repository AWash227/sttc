#define _POSIX_C_SOURCE 200809L
#include "stt/features.h"

#include <fftw3.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define STT_SAMPLE_RATE 16000
#define STT_N_FFT 512
#define STT_WIN_LENGTH 400
#define STT_HOP_LENGTH 160
#define STT_MEL_BINS 128
#define STT_PREEMPHASIS 0.97f
#define STT_LOG_ZERO_GUARD 5.960464477539063e-08f
#define STT_EPSILON 1.0e-5f
#define STT_PI 3.14159265358979323846

static float g_window[STT_WIN_LENGTH];
static float g_filters[STT_MEL_BINS * (STT_N_FFT / 2 + 1)];
static float *g_frame;
static fftwf_complex *g_spectrum;
static float *g_power;
static fftwf_plan g_fft;
static int g_tables_ready;

static double hz_to_mel(double f) {
  const double f_sp = 200.0 / 3.0;
  double m = f / f_sp;
  if (f >= 1000.0) {
    m = 15.0 + log(f / 1000.0) / log(6.4) * 27.0;
  }
  return m;
}

static double mel_to_hz(double m) {
  const double f_sp = 200.0 / 3.0;
  if (m < 15.0) return f_sp * m;
  return 1000.0 * exp(log(6.4) * (m - 15.0) / 27.0);
}

static void build_hann(float *window) {
  for (int i = 0; i < STT_WIN_LENGTH; ++i) {
    window[i] = (float)(0.5 - 0.5 * cos((2.0 * STT_PI * i) / (STT_WIN_LENGTH - 1)));
  }
}

static void build_mel_filters(float *filters) {
  memset(filters, 0, STT_MEL_BINS * (STT_N_FFT / 2 + 1) * sizeof(float));
  double mel_min = hz_to_mel(0.0);
  double mel_max = hz_to_mel(STT_SAMPLE_RATE / 2.0);
  double mel_points[STT_MEL_BINS + 2];
  double hz_points[STT_MEL_BINS + 2];
  for (int i = 0; i < STT_MEL_BINS + 2; ++i) {
    mel_points[i] = mel_min + (mel_max - mel_min) * i / (STT_MEL_BINS + 1);
    hz_points[i] = mel_to_hz(mel_points[i]);
  }
  for (int m = 0; m < STT_MEL_BINS; ++m) {
    double lower = hz_points[m];
    double center = hz_points[m + 1];
    double upper = hz_points[m + 2];
    double enorm = 2.0 / (upper - lower);
    for (int k = 0; k <= STT_N_FFT / 2; ++k) {
      double hz = (double)k * STT_SAMPLE_RATE / STT_N_FFT;
      double w = 0.0;
      if (hz >= lower && hz <= center && center > lower) {
        w = (hz - lower) / (center - lower);
      } else if (hz > center && hz <= upper && upper > center) {
        w = (upper - hz) / (upper - center);
      }
      filters[m * (STT_N_FFT / 2 + 1) + k] = (float)(w * enorm);
    }
  }
}

static float audio_sample_preemphasized(const SttAudioBuffer *audio, long idx) {
  if (idx < 0 || (size_t)idx >= audio->len) return 0.0f;
  float x = (float)audio->samples[idx] / 32768.0f;
  if (idx == 0) return x;
  float prev = (float)audio->samples[idx - 1] / 32768.0f;
  return x - STT_PREEMPHASIS * prev;
}

static void ensure_tables(void) {
  if (g_tables_ready) return;
  build_hann(g_window);
  build_mel_filters(g_filters);
  g_tables_ready = 1;
}

static int ensure_workspace(void) {
  if (g_fft) return 0;
  g_frame = fftwf_malloc(STT_N_FFT * sizeof(*g_frame));
  g_spectrum = fftwf_malloc((STT_N_FFT / 2 + 1) * sizeof(*g_spectrum));
  g_power = calloc(STT_N_FFT / 2 + 1, sizeof(*g_power));
  if (!g_frame || !g_spectrum || !g_power) goto fail;
  g_fft = fftwf_plan_dft_r2c_1d(STT_N_FFT, g_frame, g_spectrum, FFTW_ESTIMATE);
  if (!g_fft) goto fail;
  return 0;

fail:
  fftwf_free(g_frame);
  fftwf_free(g_spectrum);
  free(g_power);
  g_frame = NULL;
  g_spectrum = NULL;
  g_power = NULL;
  g_fft = NULL;
  return -1;
}

int stt_extract_features(const SttAudioBuffer *audio, SttFeatures *out) {
  memset(out, 0, sizeof(*out));
  if (!audio || !audio->samples || audio->len == 0 || audio->sample_rate != STT_SAMPLE_RATE) return -1;

  ensure_tables();
  if (ensure_workspace() != 0) return -1;

  size_t frames = audio->len / STT_HOP_LENGTH;
  if (frames == 0) frames = 1;
  out->data = malloc(frames * STT_MEL_BINS * sizeof(*out->data));
  if (!out->data) {
    stt_features_free(out);
    return -1;
  }
  out->frames = frames;
  out->valid_frames = frames;
  out->bins = STT_MEL_BINS;

  int win_pad = (STT_N_FFT - STT_WIN_LENGTH) / 2;
  for (size_t t = 0; t < frames; ++t) {
    long frame_origin = (long)(t * STT_HOP_LENGTH) - STT_N_FFT / 2;
    for (int n = 0; n < STT_N_FFT; ++n) {
      int widx = n - win_pad;
      float w = (widx >= 0 && widx < STT_WIN_LENGTH) ? g_window[widx] : 0.0f;
      g_frame[n] = audio_sample_preemphasized(audio, frame_origin + n) * w;
    }
    fftwf_execute(g_fft);
    for (int k = 0; k <= STT_N_FFT / 2; ++k) {
      float real = g_spectrum[k][0];
      float imag = g_spectrum[k][1];
      g_power[k] = (float)(real * real + imag * imag);
    }
    for (int m = 0; m < STT_MEL_BINS; ++m) {
      double v = 0.0;
      for (int k = 0; k <= STT_N_FFT / 2; ++k) {
        v += g_filters[m * (STT_N_FFT / 2 + 1) + k] * g_power[k];
      }
      out->data[m * frames + t] = logf((float)v + STT_LOG_ZERO_GUARD);
    }
  }

  for (int m = 0; m < STT_MEL_BINS; ++m) {
    double mean = 0.0;
    for (size_t t = 0; t < out->valid_frames; ++t) mean += out->data[m * frames + t];
    mean /= (double)out->valid_frames;
    double var = 0.0;
    for (size_t t = 0; t < out->valid_frames; ++t) {
      double d = out->data[m * frames + t] - mean;
      var += d * d;
    }
    var /= (double)(out->valid_frames > 1 ? out->valid_frames - 1 : 1);
    float inv_std = 1.0f / (sqrtf((float)var) + STT_EPSILON);
    for (size_t t = 0; t < frames; ++t) {
      float norm = (out->data[m * frames + t] - (float)mean) * inv_std;
      out->data[m * frames + t] = norm;
    }
  }

  return 0;
}

void stt_features_free(SttFeatures *features) {
  if (!features) return;
  free(features->data);
  memset(features, 0, sizeof(*features));
}
