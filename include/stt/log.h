#ifndef STT_LOG_H
#define STT_LOG_H

typedef enum {
  STT_LOG_ERROR = 0,
  STT_LOG_WARN = 1,
  STT_LOG_INFO = 2,
  STT_LOG_DEBUG = 3,
  STT_LOG_TRACE = 4,
} SttLogLevel;

int stt_log_enabled(SttLogLevel level);
void stt_log(SttLogLevel level, const char *fmt, ...);
long long stt_now_ms(void);

#define LOG_ERROR(...) do { if (stt_log_enabled(STT_LOG_ERROR)) stt_log(STT_LOG_ERROR, __VA_ARGS__); } while (0)
#define LOG_WARN(...) do { if (stt_log_enabled(STT_LOG_WARN)) stt_log(STT_LOG_WARN, __VA_ARGS__); } while (0)
#define LOG_INFO(...) do { if (stt_log_enabled(STT_LOG_INFO)) stt_log(STT_LOG_INFO, __VA_ARGS__); } while (0)
#define LOG_DEBUG(...) do { if (stt_log_enabled(STT_LOG_DEBUG)) stt_log(STT_LOG_DEBUG, __VA_ARGS__); } while (0)
#define LOG_TRACE(...) do { if (stt_log_enabled(STT_LOG_TRACE)) stt_log(STT_LOG_TRACE, __VA_ARGS__); } while (0)

#endif
