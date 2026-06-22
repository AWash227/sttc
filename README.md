# stt

Small C dictation CLI targeting Parakeet TDT ONNX models through ONNX Runtime.

## Status

Implemented:

- CLI command: `run`
- TDT ONNX model directory contract under `~/.models/parakeet-tdt`
- ONNX Runtime execution of TDT encoder and decoder/joint graphs
- Selectable ONNX Runtime provider request: CPU baseline, CUDA by default, and optional DirectML/CoreML/OpenVINO/MIGraphX/XNNPACK builds
- Selectable audio backend: PulseAudio, PortAudio, or disabled/headless
- Selectable hotkey backend: Linux input-device `Super+V` grabbing or disabled/headless
- Selectable text backend: XTest ASCII typing or stdout/headless
- TDT greedy decoding for logits shaped as token logits plus duration logits

Not complete yet:

- Feature extraction parity tests against Hugging Face `processor_config.json`
- Full UTF-8 text injection through X11 clipboard ownership

## Build

```sh
cmake -S . -B build
cmake --build build
```

Useful backend configurations:

```sh
# Default Linux desktop build: PulseAudio + Linux evdev/uinput hotkey + X11 typing.
cmake -S . -B build

# Headless CPU-oriented build with no PulseAudio, X11, or Linux input dependency.
cmake -S . -B build-headless \
  -DSTT_AUDIO_BACKEND=none \
  -DSTT_HOTKEY_BACKEND=none \
  -DSTT_TEXT_BACKEND=stdout \
  -DSTT_ENABLE_CUDA=OFF

# PortAudio capture with stdout output.
cmake -S . -B build-portaudio \
  -DSTT_AUDIO_BACKEND=portaudio \
  -DSTT_HOTKEY_BACKEND=none \
  -DSTT_TEXT_BACKEND=stdout
```

Provider toggles are CMake options: `STT_ENABLE_CUDA`, `STT_ENABLE_DIRECTML`, `STT_ENABLE_COREML`, `STT_ENABLE_OPENVINO`, `STT_ENABLE_MIGRAPHX`, and `STT_ENABLE_XNNPACK`.

## Usage

```sh
./build/stt run --dry-run
```

Runtime inference controls:

```sh
./build/stt run --infer-provider auto
./build/stt run --infer-provider cpu --threads 4
./build/stt run --infer-provider cuda --device-id 0
./build/stt run --model-variant fp32
./build/stt run --model-variant int8
```

`auto` tries enabled providers in build order and falls back to CPU. An explicit non-CPU provider fails if it is not enabled or unavailable in the installed ONNX Runtime package. `fp32` is the portable model variant; `int8` is kept to CPU/CUDA paths until other providers are verified.

With no subcommand, `stt` runs the dictation loop with the same defaults as `stt run`.
Use `--log file.md` to also keep a continuous microphone transcription log, one utterance per line, while preserving the normal `Super+V` dictation behavior.

`run` grabs `/dev/input/by-id/*-event-kbd` and creates a `/dev/uinput` passthrough keyboard so the hotkey works even when virt-manager has focus. If automatic keyboard discovery is wrong, set:

```sh
STT_KEYBOARD_DEVICE=/dev/input/by-id/YOUR_KEYBOARD-event-kbd ./build/stt run --dry-run
```

Logs default to `info`. Use `STT_LOG_LEVEL` for quieter output or deeper tracing:

```sh
STT_LOG_LEVEL=error ./build/stt run --dry-run
STT_LOG_LEVEL=debug ./build/stt run --dry-run
STT_LOG_LEVEL=trace ./build/stt run --dry-run
```

`info` shows user-facing startup, listening, transcription, and typed/logged transcript messages. `debug` adds performance, capture, queue, and runtime details. `trace` adds hotkey transitions and inference stage timings.

The current default dictation model directory is:

```text
~/.models/parakeet-tdt
```

The planned dictation command is:

```sh
./build/stt --type-delay-ms 0 --pre-roll-ms 350 --post-roll-ms 0
```

Typing defaults to `--type-delay-ms 0`. If a target application drops synthetic key events, add a small delay such as `--type-delay-ms 5`.
`--max-audio-sec` controls the maximum inference segment size. Longer `Super+V` holds are split into ordered segments instead of dropping speech after the limit.
Post-roll defaults to `--post-roll-ms 0`; add a small value if a setup starts clipping final words.
