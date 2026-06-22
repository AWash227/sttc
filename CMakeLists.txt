cmake_minimum_required(VERSION 3.22)
project(stt C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(STT_ENABLE_ORT "Enable ONNX Runtime inference" ON)
option(STT_ENABLE_CUDA "Enable ONNX Runtime CUDA provider selection" ON)
option(STT_ENABLE_DIRECTML "Enable ONNX Runtime DirectML provider selection" OFF)
option(STT_ENABLE_COREML "Enable ONNX Runtime CoreML provider selection" OFF)
option(STT_ENABLE_OPENVINO "Enable ONNX Runtime OpenVINO provider selection" OFF)
option(STT_ENABLE_MIGRAPHX "Enable ONNX Runtime MIGraphX provider selection" OFF)
option(STT_ENABLE_XNNPACK "Enable ONNX Runtime XNNPACK provider selection" OFF)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(STT_DEFAULT_AUDIO_BACKEND "pulse")
  set(STT_DEFAULT_HOTKEY_BACKEND "linux_evdev")
  set(STT_DEFAULT_TEXT_BACKEND "x11")
else()
  set(STT_DEFAULT_AUDIO_BACKEND "none")
  set(STT_DEFAULT_HOTKEY_BACKEND "none")
  set(STT_DEFAULT_TEXT_BACKEND "stdout")
endif()

set(STT_AUDIO_BACKEND "${STT_DEFAULT_AUDIO_BACKEND}" CACHE STRING "Audio backend: pulse, portaudio, none")
set_property(CACHE STT_AUDIO_BACKEND PROPERTY STRINGS pulse portaudio none)
set(STT_HOTKEY_BACKEND "${STT_DEFAULT_HOTKEY_BACKEND}" CACHE STRING "Hotkey backend: linux_evdev, none")
set_property(CACHE STT_HOTKEY_BACKEND PROPERTY STRINGS linux_evdev none)
set(STT_TEXT_BACKEND "${STT_DEFAULT_TEXT_BACKEND}" CACHE STRING "Text output backend: x11, stdout")
set_property(CACHE STT_TEXT_BACKEND PROPERTY STRINGS x11 stdout)

find_package(PkgConfig REQUIRED)

set(STT_SOURCES
  src/audio/recorder.c
  src/cli.c
  src/features.c
  src/fs.c
  src/infer.c
  src/log.c
  src/main.c
  src/model.c
)
set(STT_DEPS_CFLAGS_OTHER "")
set(STT_DEPS_INCLUDE_DIRS "")
set(STT_DEPS_LIBRARY_DIRS "")
set(STT_DEPS_LIBRARIES "")

function(stt_add_pkg pkg)
  pkg_check_modules(STT_PKG REQUIRED ${pkg})
  list(APPEND STT_DEPS_CFLAGS_OTHER ${STT_PKG_CFLAGS_OTHER})
  list(APPEND STT_DEPS_INCLUDE_DIRS ${STT_PKG_INCLUDE_DIRS})
  list(APPEND STT_DEPS_LIBRARY_DIRS ${STT_PKG_LIBRARY_DIRS})
  list(APPEND STT_DEPS_LIBRARIES ${STT_PKG_LIBRARIES})
  set(STT_DEPS_CFLAGS_OTHER "${STT_DEPS_CFLAGS_OTHER}" PARENT_SCOPE)
  set(STT_DEPS_INCLUDE_DIRS "${STT_DEPS_INCLUDE_DIRS}" PARENT_SCOPE)
  set(STT_DEPS_LIBRARY_DIRS "${STT_DEPS_LIBRARY_DIRS}" PARENT_SCOPE)
  set(STT_DEPS_LIBRARIES "${STT_DEPS_LIBRARIES}" PARENT_SCOPE)
endfunction()

stt_add_pkg(fftw3f)
if(STT_ENABLE_ORT)
  stt_add_pkg(libonnxruntime)
else()
  message(FATAL_ERROR "STT_ENABLE_ORT=OFF is not implemented yet; CPU ONNX Runtime is the required portability baseline")
endif()

if(STT_AUDIO_BACKEND STREQUAL "pulse")
  stt_add_pkg(libpulse-simple)
  list(APPEND STT_SOURCES src/audio/pulse.c)
elseif(STT_AUDIO_BACKEND STREQUAL "portaudio")
  stt_add_pkg(portaudio-2.0)
  list(APPEND STT_SOURCES src/audio/portaudio.c)
elseif(STT_AUDIO_BACKEND STREQUAL "none")
  list(APPEND STT_SOURCES src/audio/none.c)
else()
  message(FATAL_ERROR "unknown STT_AUDIO_BACKEND=${STT_AUDIO_BACKEND}")
endif()

if(STT_HOTKEY_BACKEND STREQUAL "linux_evdev")
  if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "STT_HOTKEY_BACKEND=linux_evdev requires Linux")
  endif()
  list(APPEND STT_SOURCES src/hotkey/linux_evdev.c)
elseif(STT_HOTKEY_BACKEND STREQUAL "none")
  list(APPEND STT_SOURCES src/hotkey/none.c)
else()
  message(FATAL_ERROR "unknown STT_HOTKEY_BACKEND=${STT_HOTKEY_BACKEND}")
endif()

if(STT_TEXT_BACKEND STREQUAL "x11")
  stt_add_pkg(x11)
  stt_add_pkg(xtst)
  list(APPEND STT_SOURCES src/text/x11.c)
elseif(STT_TEXT_BACKEND STREQUAL "stdout")
  list(APPEND STT_SOURCES src/text/stdout.c)
else()
  message(FATAL_ERROR "unknown STT_TEXT_BACKEND=${STT_TEXT_BACKEND}")
endif()

add_executable(stt ${STT_SOURCES})

target_include_directories(stt PRIVATE include)
target_include_directories(stt SYSTEM PRIVATE ${STT_DEPS_INCLUDE_DIRS})
target_compile_options(stt PRIVATE -Wall -Wextra -Wpedantic ${STT_DEPS_CFLAGS_OTHER})
target_compile_definitions(stt PRIVATE
  STT_ENABLE_CUDA=$<BOOL:${STT_ENABLE_CUDA}>
  STT_ENABLE_DIRECTML=$<BOOL:${STT_ENABLE_DIRECTML}>
  STT_ENABLE_COREML=$<BOOL:${STT_ENABLE_COREML}>
  STT_ENABLE_OPENVINO=$<BOOL:${STT_ENABLE_OPENVINO}>
  STT_ENABLE_MIGRAPHX=$<BOOL:${STT_ENABLE_MIGRAPHX}>
  STT_ENABLE_XNNPACK=$<BOOL:${STT_ENABLE_XNNPACK}>
)
target_link_directories(stt PRIVATE ${STT_DEPS_LIBRARY_DIRS})
target_link_libraries(stt PRIVATE ${STT_DEPS_LIBRARIES} m pthread)

include(CTest)
if(BUILD_TESTING)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(hotkey_tests tests/hotkey_tests.c src/log.c)
    target_include_directories(hotkey_tests PRIVATE include)
    target_compile_options(hotkey_tests PRIVATE -Wall -Wextra -Wpedantic)
    add_test(NAME hotkey_tests COMMAND hotkey_tests)
  endif()

  add_executable(audio_tests tests/audio_tests.c src/audio/recorder.c src/audio/none.c src/log.c)
  target_compile_definitions(audio_tests PRIVATE STT_TESTING)
  target_include_directories(audio_tests PRIVATE include)
  target_compile_options(audio_tests PRIVATE -Wall -Wextra -Wpedantic)
  target_link_libraries(audio_tests PRIVATE pthread)
  add_test(NAME audio_tests COMMAND audio_tests)

  add_executable(cli_tests tests/cli_tests.c src/cli.c)
  target_include_directories(cli_tests PRIVATE include)
  target_compile_options(cli_tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME cli_tests COMMAND cli_tests)

  add_executable(model_tests tests/model_tests.c src/model.c src/fs.c src/log.c)
  target_include_directories(model_tests PRIVATE include)
  target_compile_options(model_tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME model_tests COMMAND model_tests)
endif()

install(TARGETS stt RUNTIME DESTINATION bin)
