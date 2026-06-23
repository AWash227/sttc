#define WIN32_LEAN_AND_MEAN
#include "backend.h"
#include "stt/log.h"

#include <windows.h>

const char *stt_text_backend_name(void) {
  return "win32";
}

static void send_key(WORD vk, DWORD flags) {
  INPUT in;
  ZeroMemory(&in, sizeof(in));
  in.type = INPUT_KEYBOARD;
  in.ki.wVk = vk;
  in.ki.dwFlags = flags;
  SendInput(1, &in, sizeof(in));
}

int stt_text_output_write(const char *text, int delay_ms) {
  (void)delay_ms;
  if (!text) return -1;
  if (!OpenClipboard(NULL)) {
    LOG_ERROR("text: failed to open Windows clipboard\n");
    return -1;
  }
  EmptyClipboard();
  int wlen = MultiByteToWideChar(CP_UTF8, 0, text, -1, NULL, 0);
  HGLOBAL mem = wlen > 0 ? GlobalAlloc(GMEM_MOVEABLE, (size_t)wlen * sizeof(wchar_t)) : NULL;
  if (!mem) {
    CloseClipboard();
    return -1;
  }
  wchar_t *dst = GlobalLock(mem);
  MultiByteToWideChar(CP_UTF8, 0, text, -1, dst, wlen);
  GlobalUnlock(mem);
  SetClipboardData(CF_UNICODETEXT, mem);
  CloseClipboard();

  send_key(VK_CONTROL, 0);
  send_key('V', 0);
  send_key('V', KEYEVENTF_KEYUP);
  send_key(VK_CONTROL, KEYEVENTF_KEYUP);
  return 0;
}
