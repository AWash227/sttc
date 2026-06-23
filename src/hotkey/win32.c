#define WIN32_LEAN_AND_MEAN
#include "backend.h"
#include "stt/log.h"

#include <windows.h>

static SttHotkeyCallback g_callback;
static void *g_user;
static int g_alt_down;
static int g_v_down;
static int g_hotkey_down;

static LRESULT CALLBACK keyboard_proc(int code, WPARAM wparam, LPARAM lparam) {
  if (code < 0) return CallNextHookEx(NULL, code, wparam, lparam);
  KBDLLHOOKSTRUCT *ev = (KBDLLHOOKSTRUCT *)lparam;
  int down = wparam == WM_KEYDOWN || wparam == WM_SYSKEYDOWN;
  int up = wparam == WM_KEYUP || wparam == WM_SYSKEYUP;
  if (!down && !up) return CallNextHookEx(NULL, code, wparam, lparam);

  if (ev->vkCode == VK_MENU || ev->vkCode == VK_LMENU || ev->vkCode == VK_RMENU) g_alt_down = down ? 1 : 0;
  if (ev->vkCode == 'V') g_v_down = down ? 1 : 0;

  int active = g_alt_down && g_v_down;
  if (active && !g_hotkey_down) {
    g_hotkey_down = 1;
    g_callback(1, g_user);
    return 1;
  }
  if (!active && g_hotkey_down) {
    g_hotkey_down = 0;
    g_callback(0, g_user);
    return 1;
  }
  return active ? 1 : CallNextHookEx(NULL, code, wparam, lparam);
}

int stt_hotkey_loop(SttHotkeyCallback cb, void *user) {
  g_callback = cb;
  g_user = user;
  HHOOK hook = SetWindowsHookExA(WH_KEYBOARD_LL, keyboard_proc, GetModuleHandleA(NULL), 0);
  if (!hook) {
    LOG_ERROR("hotkey: failed to install Windows keyboard hook\n");
    return -1;
  }
  LOG_INFO("Hold Alt+V to dictate. Press Ctrl+C to quit.\n");
  MSG msg;
  while (GetMessageA(&msg, NULL, 0, 0) > 0) {
    TranslateMessage(&msg);
    DispatchMessageA(&msg);
  }
  UnhookWindowsHookEx(hook);
  return 0;
}
