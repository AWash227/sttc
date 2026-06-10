#ifndef STT_HOTKEY_H
#define STT_HOTKEY_H

typedef void (*SttHotkeyCallback)(int pressed, void *user);

int stt_hotkey_loop(SttHotkeyCallback cb, void *user);

#endif
