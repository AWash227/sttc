# stt

Small C dictation app targeting Parakeet TDT ONNX models through ONNX Runtime.

## Status

Implemented:

- Default command starts the dictation loop.
- First-run setup creates config/model directories, writes config if missing, checks model files, and prompts before download.
- Linux x86-64 build: PulseAudio capture, `Meta+V`, X11 typing, ONNX Runtime inference.
- Windows x86-64 build: PortAudio capture, `Ctrl+Shift`, Win32 text typing, ONNX Runtime inference, first-run download, bundled runtime DLLs.
- Runtime provider preference: `auto`, `cpu`, `cuda`, `directml`, `coreml`, `openvino`, `migraphx`, `xnnpack`.
- TDT greedy decoding for token logits plus duration logits.

Not complete yet:

- Feature extraction parity tests against Hugging Face `processor_config.json`.
- Full UTF-8 text injection through X11 clipboard ownership.
- The official Hugging Face ONNX repo id still needs final confirmation; the current downloader constant is isolated in `src/app.c`.

## Build

Linux:

```sh
cmake --preset linux-x86-64
cmake --build --preset linux-x86-64
ctest --preset linux-x86-64
```

Windows native build with MSYS2 MinGW64:

Install MSYS2, open the "MSYS2 MinGW x64" shell, then install build dependencies:

```sh
pacman -S --needed mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake mingw-w64-x86_64-ninja mingw-w64-x86_64-pkgconf mingw-w64-x86_64-portaudio mingw-w64-x86_64-fftw
```

The Windows preset expects the ONNX Runtime DirectML package at `build/deps/onnxruntime-directml-win-x64-1.24.4`. If it is in another location, pass `-DSTT_ORT_ROOT=/path/to/onnxruntime-package`.

```sh
cmake --preset windows-x86-64
cmake --build --preset windows-x86-64
ctest --preset windows-x86-64
```

Release artifact names should be:

- `linux-x86-64`
- `windows-x86-64`

## Usage

```sh
./build/linux-x86-64/stt --dry-run
```

On first run, `stt` creates config/model directories, writes a default config if needed, checks the model bundle, and asks before downloading missing model files.

Default paths:

- Linux config: `~/.config/stt/config.toml`
- Linux model: `~/.models/parakeet-tdt`
- Windows config: `%APPDATA%\stt\config.toml`
- Windows model: `%LOCALAPPDATA%\stt\models\parakeet-tdt`

Override hooks:

```sh
STT_CONFIG_HOME=/tmp/stt-config STT_MODEL_DIR=/tmp/parakeet ./build/linux-x86-64/stt --dry-run
```

Runtime inference controls:

```sh
./build/linux-x86-64/stt --infer-provider auto
./build/linux-x86-64/stt --infer-provider cpu --threads 4
./build/linux-x86-64/stt --infer-provider cuda --device-id 0
./build/linux-x86-64/stt --model-variant fp32
./build/linux-x86-64/stt --model-variant int8
```

`auto` tries enabled providers in build order and falls back to CPU. An explicit non-CPU provider fails if it is not enabled or unavailable in the installed ONNX Runtime package.

On Linux, `stt` grabs `/dev/input/by-id/*-event-kbd` and creates a `/dev/uinput` passthrough keyboard so `Meta+V` works even when virt-manager has focus. If automatic keyboard discovery is wrong, set:

```sh
STT_KEYBOARD_DEVICE=/dev/input/by-id/YOUR_KEYBOARD-event-kbd ./build/linux-x86-64/stt --dry-run
```

On Windows, the default hotkey is `Ctrl+Shift`.

Logs default to `info`. Use `STT_LOG_LEVEL` for quieter output or deeper tracing:

```sh
STT_LOG_LEVEL=error ./build/linux-x86-64/stt --dry-run
STT_LOG_LEVEL=debug ./build/linux-x86-64/stt --dry-run
STT_LOG_LEVEL=trace ./build/linux-x86-64/stt --dry-run
```

Typing defaults to `--type-delay-ms 0`. If a target application drops synthetic key events, add a small delay such as `--type-delay-ms 5`.
`--max-audio-sec` controls the maximum inference segment size. Longer hotkey holds are split into ordered segments instead of dropping speech after the limit.
Post-roll defaults to `--post-roll-ms 0`; add a small value if a setup starts clipping final words.
