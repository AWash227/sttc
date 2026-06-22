#include "stt/log.h"

#include "backend.h"

int stt_hotkey_loop(SttHotkeyCallback cb, void *user) {
  (void)cb;
  (void)user;
  LOG_ERROR("hotkey backend is disabled in this build\n");
  return -1;
}
