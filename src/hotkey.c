#define _POSIX_C_SOURCE 200809L
#include "stt/hotkey.h"
#include "stt/log.h"

#include <errno.h>
#include <fcntl.h>
#include <glob.h>
#include <linux/input.h>
#include <linux/uinput.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/time.h>
#include <unistd.h>

#define MAX_KEY_CODE 255

typedef struct {
  int fd;
  char *path;
} Keyboard;

typedef struct {
  Keyboard *items;
  size_t len;
} Keyboards;

typedef struct {
  int super_l_down;
  int super_r_down;
  int super_l_emitted;
  int super_r_emitted;
  int super_l_consumed;
  int super_r_consumed;
  int v_down;
  int hotkey_down;
} HotkeyState;

static int is_super_key(unsigned short code) {
  return code == KEY_LEFTMETA || code == KEY_RIGHTMETA;
}

static int emit_event(int fd, unsigned short type, unsigned short code, int value) {
  struct input_event ev;
  memset(&ev, 0, sizeof(ev));
  gettimeofday(&ev.time, NULL);
  ev.type = type;
  ev.code = code;
  ev.value = value;
  return write(fd, &ev, sizeof(ev)) == (ssize_t)sizeof(ev) ? 0 : -1;
}

static int emit_key(int fd, unsigned short code, int value) {
  if (emit_event(fd, EV_KEY, code, value) != 0) return -1;
  return emit_event(fd, EV_SYN, SYN_REPORT, 0);
}

static int forward_event(int fd, const struct input_event *ev) {
  return write(fd, ev, sizeof(*ev)) == (ssize_t)sizeof(*ev) ? 0 : -1;
}

static int open_virtual_keyboard(void) {
  int fd = open("/dev/uinput", O_WRONLY | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) {
    LOG_ERROR("hotkey: open /dev/uinput failed: %s\n", strerror(errno));
    return -1;
  }

  if (ioctl(fd, UI_SET_EVBIT, EV_KEY) < 0 ||
      ioctl(fd, UI_SET_EVBIT, EV_SYN) < 0 ||
      ioctl(fd, UI_SET_EVBIT, EV_REP) < 0) {
    LOG_ERROR("hotkey: configure /dev/uinput failed: %s\n", strerror(errno));
    close(fd);
    return -1;
  }

  for (int code = 1; code <= MAX_KEY_CODE; ++code) {
    ioctl(fd, UI_SET_KEYBIT, code);
  }

  struct uinput_setup setup;
  memset(&setup, 0, sizeof(setup));
  snprintf(setup.name, sizeof(setup.name), "stt hotkey passthrough");
  setup.id.bustype = BUS_USB;
  setup.id.vendor = 0x5354;
  setup.id.product = 0x5454;
  setup.id.version = 1;

  if (ioctl(fd, UI_DEV_SETUP, &setup) < 0 || ioctl(fd, UI_DEV_CREATE) < 0) {
    LOG_ERROR("hotkey: create virtual keyboard failed: %s\n", strerror(errno));
    close(fd);
    return -1;
  }

  return fd;
}

static void close_keyboards(Keyboards *keyboards) {
  for (size_t i = 0; i < keyboards->len; ++i) {
    if (keyboards->items[i].fd >= 0) {
      ioctl(keyboards->items[i].fd, EVIOCGRAB, 0);
      close(keyboards->items[i].fd);
    }
    free(keyboards->items[i].path);
  }
  free(keyboards->items);
  memset(keyboards, 0, sizeof(*keyboards));
}

static int add_keyboard(Keyboards *keyboards, const char *path) {
  Keyboard *next = realloc(keyboards->items, (keyboards->len + 1) * sizeof(*keyboards->items));
  if (!next) return -1;
  keyboards->items = next;
  keyboards->items[keyboards->len] = (Keyboard){.fd = -1, .path = strdup(path)};
  if (!keyboards->items[keyboards->len].path) return -1;
  keyboards->len++;
  return 0;
}

static int discover_keyboards(Keyboards *keyboards) {
  memset(keyboards, 0, sizeof(*keyboards));

  const char *device = getenv("STT_KEYBOARD_DEVICE");
  if (device && *device) return add_keyboard(keyboards, device);

  glob_t matches;
  memset(&matches, 0, sizeof(matches));
  if (glob("/dev/input/by-id/*-event-kbd", 0, NULL, &matches) != 0 || matches.gl_pathc == 0) {
    globfree(&matches);
    LOG_ERROR("hotkey: no keyboard devices found; set STT_KEYBOARD_DEVICE=/dev/input/...\n");
    return -1;
  }

  for (size_t i = 0; i < matches.gl_pathc; ++i) {
    if (add_keyboard(keyboards, matches.gl_pathv[i]) != 0) {
      globfree(&matches);
      close_keyboards(keyboards);
      return -1;
    }
  }
  globfree(&matches);
  return 0;
}

static int grab_keyboards(Keyboards *keyboards) {
  size_t grabbed = 0;
  for (size_t i = 0; i < keyboards->len; ++i) {
    Keyboard *keyboard = &keyboards->items[i];
    keyboard->fd = open(keyboard->path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (keyboard->fd < 0) {
      LOG_WARN("hotkey: open %s failed: %s\n", keyboard->path, strerror(errno));
      continue;
    }
    if (ioctl(keyboard->fd, EVIOCGRAB, 1) < 0) {
      LOG_WARN("hotkey: grab %s failed: %s\n", keyboard->path, strerror(errno));
      close(keyboard->fd);
      keyboard->fd = -1;
      continue;
    }
    grabbed++;
  }

  if (grabbed == 0) {
    LOG_ERROR("hotkey: failed to grab any keyboard device\n");
    return -1;
  }

  LOG_INFO("hotkey: grabbed_keyboards=%zu passthrough=uinput hotkey=Super+V\n", grabbed);
  for (size_t i = 0; i < keyboards->len; ++i) {
    if (keyboards->items[i].fd >= 0) LOG_DEBUG("hotkey: keyboard=%s\n", keyboards->items[i].path);
  }
  return 0;
}

static void release_emitted_super_keys(int uinput_fd, HotkeyState *state) {
  if (state->super_l_emitted) {
    emit_key(uinput_fd, KEY_LEFTMETA, 0);
    state->super_l_emitted = 0;
  }
  if (state->super_r_emitted) {
    emit_key(uinput_fd, KEY_RIGHTMETA, 0);
    state->super_r_emitted = 0;
  }
}

static void flush_pending_super_keys(int uinput_fd, HotkeyState *state) {
  if (state->super_l_down && !state->super_l_emitted) {
    emit_key(uinput_fd, KEY_LEFTMETA, 1);
    state->super_l_emitted = 1;
  }
  if (state->super_r_down && !state->super_r_emitted) {
    emit_key(uinput_fd, KEY_RIGHTMETA, 1);
    state->super_r_emitted = 1;
  }
}

static void handle_super_release(int uinput_fd, HotkeyState *state, unsigned short code) {
  int *consumed = code == KEY_LEFTMETA ? &state->super_l_consumed : &state->super_r_consumed;
  int *emitted = code == KEY_LEFTMETA ? &state->super_l_emitted : &state->super_r_emitted;

  if (*consumed) {
    *consumed = 0;
  } else if (*emitted) {
    emit_key(uinput_fd, code, 0);
    *emitted = 0;
  } else {
    emit_key(uinput_fd, code, 1);
    emit_key(uinput_fd, code, 0);
  }
}

static int handle_key_event(int uinput_fd, HotkeyState *state, const struct input_event *ev, SttHotkeyCallback cb, void *user) {
  int pass = 1;
  int was_hotkey_down = state->hotkey_down;

  if (ev->code == KEY_LEFTMETA) state->super_l_down = ev->value != 0;
  if (ev->code == KEY_RIGHTMETA) state->super_r_down = ev->value != 0;
  if (ev->code == KEY_V) state->v_down = ev->value != 0;

  int super_down = state->super_l_down || state->super_r_down;
  if (!state->hotkey_down && ev->code == KEY_V && ev->value == 1 && super_down) {
    state->hotkey_down = 1;
    pass = 0;
    release_emitted_super_keys(uinput_fd, state);
    state->super_l_consumed = state->super_l_down;
    state->super_r_consumed = state->super_r_down;
    LOG_TRACE("hotkey: combo=down\n");
    cb(1, user);
  } else if (state->hotkey_down && (ev->code == KEY_V || is_super_key(ev->code))) {
    pass = 0;
  } else if (!state->hotkey_down && is_super_key(ev->code)) {
    pass = 0;
    if (ev->value == 0) handle_super_release(uinput_fd, state, ev->code);
  } else if (!state->hotkey_down && ev->value != 0) {
    flush_pending_super_keys(uinput_fd, state);
  }

  super_down = state->super_l_down || state->super_r_down;
  if (was_hotkey_down && (!super_down || !state->v_down)) {
    state->hotkey_down = 0;
    LOG_TRACE("hotkey: combo=up\n");
    cb(0, user);
  }

  return pass;
}

int stt_hotkey_loop(SttHotkeyCallback cb, void *user) {
  Keyboards keyboards;
  if (discover_keyboards(&keyboards) != 0) return -1;
  if (grab_keyboards(&keyboards) != 0) {
    close_keyboards(&keyboards);
    return -1;
  }

  int uinput_fd = open_virtual_keyboard();
  if (uinput_fd < 0) {
    close_keyboards(&keyboards);
    return -1;
  }

  struct pollfd *pollfds = calloc(keyboards.len, sizeof(*pollfds));
  if (!pollfds) {
    ioctl(uinput_fd, UI_DEV_DESTROY);
    close(uinput_fd);
    close_keyboards(&keyboards);
    return -1;
  }
  for (size_t i = 0; i < keyboards.len; ++i) pollfds[i].fd = keyboards.items[i].fd;

  HotkeyState state = {0};
  for (;;) {
    for (size_t i = 0; i < keyboards.len; ++i) {
      pollfds[i].events = keyboards.items[i].fd >= 0 ? POLLIN : 0;
      pollfds[i].revents = 0;
    }

    int rc = poll(pollfds, keyboards.len, -1);
    if (rc < 0) {
      if (errno == EINTR) continue;
      LOG_ERROR("hotkey: poll failed: %s\n", strerror(errno));
      break;
    }

    for (size_t i = 0; i < keyboards.len; ++i) {
      if (!(pollfds[i].revents & POLLIN)) continue;

      for (;;) {
        struct input_event ev;
        ssize_t n = read(keyboards.items[i].fd, &ev, sizeof(ev));
        if (n < 0) {
          if (errno == EINTR) continue;
          if (errno == EAGAIN || errno == EWOULDBLOCK) break;
          LOG_ERROR("hotkey: read %s failed: %s\n", keyboards.items[i].path, strerror(errno));
          break;
        }
        if (n != (ssize_t)sizeof(ev)) break;

        int pass = 1;
        if (ev.type == EV_KEY) {
          pass = handle_key_event(uinput_fd, &state, &ev, cb, user);
        } else if (ev.type != EV_SYN && ev.type != EV_REP) {
          pass = 0;
        }

        if (pass) forward_event(uinput_fd, &ev);
      }
    }
  }

  free(pollfds);
  ioctl(uinput_fd, UI_DEV_DESTROY);
  close(uinput_fd);
  close_keyboards(&keyboards);
  return -1;
}
