#ifndef STT_HOTKEY_BACKEND_INTERNAL_H
#define STT_HOTKEY_BACKEND_INTERNAL_H

typedef void (*SttHotkeyCallback)(int pressed, void *user);

int stt_hotkey_loop(SttHotkeyCallback cb, void *user);

#endif
