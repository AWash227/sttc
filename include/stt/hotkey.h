#ifndef STT_HOTKEY_H
#define STT_HOTKEY_H

typedef void (*SttHotkeyCallback)(int pressed, void *user);

int stt_hotkey_loop(const char *hotkey, SttHotkeyCallback cb, void *user);

#endif
