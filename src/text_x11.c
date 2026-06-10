#define _POSIX_C_SOURCE 200809L
#include "stt/text_x11.h"

#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <X11/extensions/XTest.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static KeySym keysym_for_ascii(char c, int *shift) {
  *shift = 0;
  if (c >= 'a' && c <= 'z') return (KeySym)c;
  if (c >= 'A' && c <= 'Z') {
    *shift = 1;
    return (KeySym)tolower((unsigned char)c);
  }
  if (c >= '0' && c <= '9') return (KeySym)c;
  switch (c) {
    case ' ': return XK_space;
    case '\n': return XK_Return;
    case '\t': return XK_Tab;
    case '.': return XK_period;
    case ',': return XK_comma;
    case ';': return XK_semicolon;
    case ':': *shift = 1; return XK_semicolon;
    case '\'': return XK_apostrophe;
    case '"': *shift = 1; return XK_apostrophe;
    case '/': return XK_slash;
    case '?': *shift = 1; return XK_slash;
    case '-': return XK_minus;
    case '_': *shift = 1; return XK_minus;
    case '=': return XK_equal;
    case '+': *shift = 1; return XK_equal;
    case '!': *shift = 1; return XK_1;
    case '@': *shift = 1; return XK_2;
    case '#': *shift = 1; return XK_3;
    case '$': *shift = 1; return XK_4;
    case '%': *shift = 1; return XK_5;
    case '^': *shift = 1; return XK_6;
    case '&': *shift = 1; return XK_7;
    case '*': *shift = 1; return XK_8;
    case '(': *shift = 1; return XK_9;
    case ')': *shift = 1; return XK_0;
    default: return NoSymbol;
  }
}

int stt_type_text_x11(const char *text, int delay_ms) {
  Display *dpy = XOpenDisplay(NULL);
  if (!dpy) {
    fprintf(stderr, "failed to open X display\n");
    return -1;
  }
  int event_base = 0, error_base = 0, major = 0, minor = 0;
  if (!XTestQueryExtension(dpy, &event_base, &error_base, &major, &minor)) {
    fprintf(stderr, "XTest extension is unavailable\n");
    XCloseDisplay(dpy);
    return -1;
  }

  KeyCode shift = XKeysymToKeycode(dpy, XK_Shift_L);
  for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
    if (*p >= 0x80) {
      fprintf(stderr, "non-ASCII output is not typed by this build; use --print for UTF-8 text\n");
      XCloseDisplay(dpy);
      return -1;
    }
    int need_shift = 0;
    KeySym sym = keysym_for_ascii((char)*p, &need_shift);
    if (sym == NoSymbol) continue;
    KeyCode code = XKeysymToKeycode(dpy, sym);
    if (!code) continue;
    if (need_shift) XTestFakeKeyEvent(dpy, shift, True, CurrentTime);
    XTestFakeKeyEvent(dpy, code, True, CurrentTime);
    XTestFakeKeyEvent(dpy, code, False, CurrentTime);
    if (need_shift) XTestFakeKeyEvent(dpy, shift, False, CurrentTime);
    XFlush(dpy);
    if (delay_ms > 0) {
      struct timespec ts;
      ts.tv_sec = delay_ms / 1000;
      ts.tv_nsec = (long)(delay_ms % 1000) * 1000000L;
      nanosleep(&ts, NULL);
    }
  }
  XCloseDisplay(dpy);
  return 0;
}
