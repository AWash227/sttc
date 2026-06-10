# stt

Small C dictation CLI targeting NVIDIA Parakeet TDT ONNX models on Linux.

## Status

Implemented:

- CLI commands: `install-model`, `inspect-model`, `test-gpu`, `run`
- TDT ONNX model directory contract under `~/.models/parakeet-tdt`
- v3 safetensors inspection under `~/.models/parakeet-tdt-0.6b-v3`
- Hugging Face file download with libcurl
- safetensors header parsing without the safetensors library
- tokenizer loading and basic Metaspace decode without the tokenizers library
- CUDA driver initialization and upload of v3 F32/BF16/F16 tensors into VRAM
- ONNX Runtime execution of TDT encoder and decoder/joint graphs
- PulseAudio default microphone capture while `Super+V` is held
- Linux input-device hotkey grabbing for `Super+V`
- XTest ASCII typing with configurable per-character delay
- TDT greedy decoder helper for logits shaped as token logits plus duration logits

Not complete yet:

- Feature extraction parity tests against Hugging Face `processor_config.json`
- Full UTF-8 text injection through X11 clipboard ownership

## Build

```sh
cmake -S . -B build
cmake --build build
```

## Usage

```sh
./build/stt install-model
./build/stt inspect-model
./build/stt test-gpu
./build/stt run --dry-run
```

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

`info` includes transcripts and compact `perf:` summaries. `debug` adds capture and queue lifecycle details. `trace` adds hotkey transitions and inference stage timings.

The current default dictation model directory is:

```text
~/.models/parakeet-tdt
```

The planned dictation command is:

```sh
./build/stt run --hotkey Super+V --type-delay-ms 0 --pre-roll-ms 350 --post-roll-ms 0
```

Typing defaults to `--type-delay-ms 0`. If a target application drops synthetic key events, add a small delay such as `--type-delay-ms 5`.
Post-roll defaults to `--post-roll-ms 0`; add a small value if a setup starts clipping final words.
