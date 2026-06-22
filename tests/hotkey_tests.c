#define _POSIX_C_SOURCE 200809L
#define STT_TESTING
#include "../src/hotkey/linux_evdev.c"

#include <assert.h>

typedef struct {
  int pressed_count;
  int released_count;
} CallbackState;

static void hotkey_cb(int pressed, void *user) {
  CallbackState *state = user;
  if (pressed) state->pressed_count++;
  else state->released_count++;
}

static struct input_event key_event(unsigned short code, int value) {
  struct input_event ev;
  memset(&ev, 0, sizeof(ev));
  ev.type = EV_KEY;
  ev.code = code;
  ev.value = value;
  return ev;
}

static void feed_event(int fd, HotkeyState *state, struct input_event ev, CallbackState *cb_state) {
  int pass = handle_key_event(fd, state, &ev, hotkey_cb, cb_state);
  if (pass) assert(forward_event(fd, &ev) == 0);
}

static void count_meta_events(int fd, int *downs, int *ups) {
  for (;;) {
    struct input_event ev;
    ssize_t n = read(fd, &ev, sizeof(ev));
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK) break;
      assert(0);
    }
    if (n == 0) break;
    assert(n == (ssize_t)sizeof(ev));
    if (ev.type == EV_KEY && ev.code == KEY_LEFTMETA && ev.value == 1) (*downs)++;
    if (ev.type == EV_KEY && ev.code == KEY_LEFTMETA && ev.value == 0) (*ups)++;
  }
}

static void test_super_release_after_forwarded_shortcut(void) {
  int pipe_fds[2];
  assert(pipe(pipe_fds) == 0);
  assert(set_nonblocking_cloexec(pipe_fds[0]) == 0);

  HotkeyState state = {0};
  CallbackState cb_state = {0};

  feed_event(pipe_fds[1], &state, key_event(KEY_LEFTMETA, 1), &cb_state);
  feed_event(pipe_fds[1], &state, key_event(KEY_V, 1), &cb_state);
  feed_event(pipe_fds[1], &state, key_event(KEY_V, 0), &cb_state);
  feed_event(pipe_fds[1], &state, key_event(KEY_A, 1), &cb_state);
  feed_event(pipe_fds[1], &state, key_event(KEY_A, 0), &cb_state);
  feed_event(pipe_fds[1], &state, key_event(KEY_LEFTMETA, 0), &cb_state);

  assert(cb_state.pressed_count == 1);
  assert(cb_state.released_count == 1);
  int downs = 0;
  int ups = 0;
  count_meta_events(pipe_fds[0], &downs, &ups);
  assert(downs == 1);
  assert(ups == 1);

  close(pipe_fds[0]);
  close(pipe_fds[1]);
}

int main(void) {
  test_super_release_after_forwarded_shortcut();
  return 0;
}
