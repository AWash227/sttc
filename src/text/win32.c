#define WIN32_LEAN_AND_MEAN
#include "backend.h"

#include <windows.h>
#include <ctype.h>
#include <stdlib.h>
#include <string.h>

const char *stt_text_backend_name(void) {
  return "win32";
}

static void sleep_ms(int delay_ms) {
  if (delay_ms > 0) Sleep((DWORD)delay_ms);
}

static int send_input(INPUT *inputs, UINT count) {
  return SendInput(count, inputs, sizeof(inputs[0])) == count ? 0 : -1;
}

static int send_key(WORD vk, DWORD flags) {
  INPUT in;
  ZeroMemory(&in, sizeof(in));
  in.type = INPUT_KEYBOARD;
  in.ki.wVk = vk;
  in.ki.dwFlags = flags;
  return send_input(&in, 1);
}

static int send_unicode_char(wchar_t ch) {
  INPUT in[2];
  ZeroMemory(in, sizeof(in));
  in[0].type = INPUT_KEYBOARD;
  in[0].ki.wScan = ch;
  in[0].ki.dwFlags = KEYEVENTF_UNICODE;
  in[1] = in[0];
  in[1].ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP;
  return send_input(in, 2);
}

static void release_modifiers(void) {
  send_key(VK_MENU, KEYEVENTF_KEYUP);
  send_key(VK_CONTROL, KEYEVENTF_KEYUP);
  send_key(VK_SHIFT, KEYEVENTF_KEYUP);
  send_key(VK_LWIN, KEYEVENTF_KEYUP);
  send_key(VK_RWIN, KEYEVENTF_KEYUP);
}

static int send_key_combo(WORD vk, BYTE shift_state) {
  INPUT in[8];
  UINT n = 0;
  ZeroMemory(in, sizeof(in));
  if (shift_state & 1) {
    in[n].type = INPUT_KEYBOARD;
    in[n++].ki.wVk = VK_SHIFT;
  }
  if (shift_state & 2) {
    in[n].type = INPUT_KEYBOARD;
    in[n++].ki.wVk = VK_CONTROL;
  }
  if (shift_state & 4) {
    in[n].type = INPUT_KEYBOARD;
    in[n++].ki.wVk = VK_MENU;
  }
  in[n].type = INPUT_KEYBOARD;
  in[n++].ki.wVk = vk;
  in[n].type = INPUT_KEYBOARD;
  in[n].ki.wVk = vk;
  in[n++].ki.dwFlags = KEYEVENTF_KEYUP;
  if (shift_state & 4) {
    in[n].type = INPUT_KEYBOARD;
    in[n].ki.wVk = VK_MENU;
    in[n++].ki.dwFlags = KEYEVENTF_KEYUP;
  }
  if (shift_state & 2) {
    in[n].type = INPUT_KEYBOARD;
    in[n].ki.wVk = VK_CONTROL;
    in[n++].ki.dwFlags = KEYEVENTF_KEYUP;
  }
  if (shift_state & 1) {
    in[n].type = INPUT_KEYBOARD;
    in[n].ki.wVk = VK_SHIFT;
    in[n++].ki.dwFlags = KEYEVENTF_KEYUP;
  }
  return send_input(in, n);
}

static int type_char(wchar_t ch) {
  SHORT mapped = VkKeyScanW(ch);
  if (mapped != -1) {
    WORD vk = (WORD)(mapped & 0xff);
    BYTE shift_state = (BYTE)((mapped >> 8) & 0xff);
    if ((shift_state & ~7) == 0) return send_key_combo(vk, shift_state);
  }
  return send_unicode_char(ch);
}

static int send_text(const wchar_t *text, int delay_ms) {
  release_modifiers();
  Sleep(30);
  for (const wchar_t *p = text; *p; ++p) {
    if (type_char(*p) != 0) return -1;
    sleep_ms(delay_ms);
  }
  return 0;
}

int stt_text_output_write(const char *text, int delay_ms) {
  if (!text) return -1;

  int wlen = MultiByteToWideChar(CP_UTF8, 0, text, -1, NULL, 0);
  wchar_t *wide = wlen > 0 ? malloc((size_t)wlen * sizeof(*wide)) : NULL;
  if (!wide) return -1;
  if (MultiByteToWideChar(CP_UTF8, 0, text, -1, wide, wlen) <= 0) {
    free(wide);
    return -1;
  }

  int rc = send_text(wide, delay_ms);
  size_t len = strlen(text);
  if (rc == 0 && len > 0 && !isspace((unsigned char)text[len - 1])) {
    rc = type_char(L' ');
  }
  free(wide);
  return rc;
}
