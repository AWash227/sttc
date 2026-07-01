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
option(STT_ENABLE_DOWNLOAD "Enable first-run Hugging Face model download" ON)
set(STT_ORT_ROOT "" CACHE PATH "Path to an ONNX Runtime binary package")

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(STT_DEFAULT_AUDIO_BACKEND "pulse")
  set(STT_DEFAULT_HOTKEY_BACKEND "linux_evdev")
  set(STT_DEFAULT_TEXT_BACKEND "x11")
elseif(WIN32)
  set(STT_DEFAULT_AUDIO_BACKEND "portaudio")
  set(STT_DEFAULT_HOTKEY_BACKEND "win32")
  set(STT_DEFAULT_TEXT_BACKEND "win32")
else()
  set(STT_DEFAULT_AUDIO_BACKEND "none")
  set(STT_DEFAULT_HOTKEY_BACKEND "none")
  set(STT_DEFAULT_TEXT_BACKEND "stdout")
endif()

set(STT_AUDIO_BACKEND "${STT_DEFAULT_AUDIO_BACKEND}" CACHE STRING "Audio backend: pulse, portaudio, none")
set_property(CACHE STT_AUDIO_BACKEND PROPERTY STRINGS pulse portaudio none)
set(STT_HOTKEY_BACKEND "${STT_DEFAULT_HOTKEY_BACKEND}" CACHE STRING "Hotkey backend: linux_evdev, none")
set_property(CACHE STT_HOTKEY_BACKEND PROPERTY STRINGS linux_evdev win32 none)
set(STT_TEXT_BACKEND "${STT_DEFAULT_TEXT_BACKEND}" CACHE STRING "Text output backend: x11, win32, stdout")
set_property(CACHE STT_TEXT_BACKEND PROPERTY STRINGS x11 win32 stdout)

find_package(PkgConfig REQUIRED)
find_package(Threads REQUIRED)

set(STT_SOURCES
  src/app.c
  src/audio/recorder.c
  src/cli.c
  src/fs.c
  src/log.c
  src/main.c
  src/model.c
)
set(STT_DEPS_CFLAGS_OTHER "")
set(STT_DEPS_INCLUDE_DIRS "")
set(STT_DEPS_LIBRARY_DIRS "")
set(STT_DEPS_LIBRARIES "")

if(WIN32)
  get_filename_component(STT_MINGW_BIN_DIR "${CMAKE_C_COMPILER}" DIRECTORY)
  foreach(STT_MINGW_DLL libwinpthread-1.dll libgcc_s_seh-1.dll)
    unset(STT_MINGW_DLL_PATH CACHE)
    find_file(STT_MINGW_DLL_PATH ${STT_MINGW_DLL}
      PATHS "${STT_MINGW_BIN_DIR}" "/usr/x86_64-w64-mingw32/bin"
      NO_DEFAULT_PATH)
    if(STT_MINGW_DLL_PATH)
      file(COPY "${STT_MINGW_DLL_PATH}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
    endif()
  endforeach()
endif()

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

function(stt_enable_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

if(STT_ENABLE_ORT)
  stt_add_pkg(fftw3f)
  if(STT_ORT_ROOT)
    if(EXISTS "${STT_ORT_ROOT}/build/native/include")
      list(APPEND STT_DEPS_INCLUDE_DIRS "${STT_ORT_ROOT}/build/native/include")
    else()
      list(APPEND STT_DEPS_INCLUDE_DIRS "${STT_ORT_ROOT}/include")
    endif()
    if(EXISTS "${STT_ORT_ROOT}/runtimes/win-x64/native")
      list(APPEND STT_DEPS_LIBRARY_DIRS "${STT_ORT_ROOT}/runtimes/win-x64/native")
      set(STT_ORT_DLL_DIR "${STT_ORT_ROOT}/runtimes/win-x64/native")
    else()
      list(APPEND STT_DEPS_LIBRARY_DIRS "${STT_ORT_ROOT}/lib")
      set(STT_ORT_DLL_DIR "${STT_ORT_ROOT}/lib")
    endif()
    list(APPEND STT_DEPS_LIBRARIES onnxruntime)
  else()
    stt_add_pkg(libonnxruntime)
  endif()
  list(APPEND STT_SOURCES src/features.c src/infer.c)
else()
  list(APPEND STT_SOURCES src/infer_stub.c)
endif()

set(STT_HAVE_CUDA_TOOLKIT FALSE)
if(STT_ENABLE_ORT AND STT_ENABLE_CUDA)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    set(STT_HAVE_CUDA_TOOLKIT TRUE)
  else()
    message(STATUS "CUDA toolkit not found; decoder CUDA graph capture will be disabled at build time")
  endif()
endif()

if(STT_ENABLE_DOWNLOAD AND NOT WIN32)
  stt_add_pkg(libcurl)
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
elseif(STT_HOTKEY_BACKEND STREQUAL "win32")
  if(NOT WIN32)
    message(FATAL_ERROR "STT_HOTKEY_BACKEND=win32 requires Windows")
  endif()
  list(APPEND STT_SOURCES src/hotkey/win32.c)
else()
  message(FATAL_ERROR "unknown STT_HOTKEY_BACKEND=${STT_HOTKEY_BACKEND}")
endif()

if(STT_TEXT_BACKEND STREQUAL "x11")
  stt_add_pkg(x11)
  stt_add_pkg(xtst)
  list(APPEND STT_SOURCES src/text/x11.c)
elseif(STT_TEXT_BACKEND STREQUAL "stdout")
  list(APPEND STT_SOURCES src/text/stdout.c)
elseif(STT_TEXT_BACKEND STREQUAL "win32")
  if(NOT WIN32)
    message(FATAL_ERROR "STT_TEXT_BACKEND=win32 requires Windows")
  endif()
  list(APPEND STT_SOURCES src/text/win32.c)
else()
  message(FATAL_ERROR "unknown STT_TEXT_BACKEND=${STT_TEXT_BACKEND}")
endif()

add_executable(stt ${STT_SOURCES})

target_include_directories(stt PRIVATE include)
target_include_directories(stt SYSTEM PRIVATE ${STT_DEPS_INCLUDE_DIRS})
stt_enable_warnings(stt)
target_compile_options(stt PRIVATE ${STT_DEPS_CFLAGS_OTHER})
target_compile_definitions(stt PRIVATE
  STT_ENABLE_ORT=$<BOOL:${STT_ENABLE_ORT}>
  STT_ENABLE_DOWNLOAD=$<BOOL:${STT_ENABLE_DOWNLOAD}>
  STT_ENABLE_CUDA=$<BOOL:${STT_ENABLE_CUDA}>
  STT_ENABLE_DIRECTML=$<BOOL:${STT_ENABLE_DIRECTML}>
  STT_ENABLE_COREML=$<BOOL:${STT_ENABLE_COREML}>
  STT_ENABLE_OPENVINO=$<BOOL:${STT_ENABLE_OPENVINO}>
  STT_ENABLE_MIGRAPHX=$<BOOL:${STT_ENABLE_MIGRAPHX}>
  STT_ENABLE_XNNPACK=$<BOOL:${STT_ENABLE_XNNPACK}>
  STT_HAVE_CUDA_TOOLKIT=$<BOOL:${STT_HAVE_CUDA_TOOLKIT}>
)
target_link_directories(stt PRIVATE ${STT_DEPS_LIBRARY_DIRS})
target_link_libraries(stt PRIVATE ${STT_DEPS_LIBRARIES} Threads::Threads)
if(STT_HAVE_CUDA_TOOLKIT)
  target_link_libraries(stt PRIVATE CUDA::cudart)
endif()
if(NOT MSVC)
  target_link_libraries(stt PRIVATE m)
endif()
if(WIN32)
  target_link_libraries(stt PRIVATE shell32 ole32 user32)
  if(STT_ENABLE_DOWNLOAD)
    target_link_libraries(stt PRIVATE winhttp)
  endif()
  if(STT_AUDIO_BACKEND STREQUAL "portaudio")
    find_file(STT_PORTAUDIO_DLL libportaudio.dll
      PATHS "${STT_MINGW_BIN_DIR}"
      NO_DEFAULT_PATH)
    if(STT_PORTAUDIO_DLL)
      add_custom_command(TARGET stt POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${STT_PORTAUDIO_DLL}"
          "$<TARGET_FILE_DIR:stt>/libportaudio.dll")
    endif()
  endif()
  find_file(STT_FFTW_DLL libfftw3f-3.dll
    PATHS "${STT_MINGW_BIN_DIR}"
    NO_DEFAULT_PATH)
  if(STT_FFTW_DLL)
    add_custom_command(TARGET stt POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${STT_FFTW_DLL}"
        "$<TARGET_FILE_DIR:stt>/libfftw3f-3.dll")
  endif()
  if(STT_ORT_ROOT)
    add_custom_command(TARGET stt POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${STT_ORT_DLL_DIR}/onnxruntime.dll"
        "$<TARGET_FILE_DIR:stt>/onnxruntime.dll"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${STT_ORT_DLL_DIR}/onnxruntime_providers_shared.dll"
        "$<TARGET_FILE_DIR:stt>/onnxruntime_providers_shared.dll")
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_executable(hotkey_tests tests/hotkey_tests.c src/log.c)
    target_include_directories(hotkey_tests PRIVATE include)
    stt_enable_warnings(hotkey_tests)
    add_test(NAME hotkey_tests COMMAND hotkey_tests)
  endif()

  add_executable(audio_tests tests/audio_tests.c src/audio/recorder.c src/audio/none.c src/log.c)
  target_compile_definitions(audio_tests PRIVATE STT_TESTING)
  target_include_directories(audio_tests PRIVATE include)
  stt_enable_warnings(audio_tests)
  target_link_libraries(audio_tests PRIVATE Threads::Threads)
  add_test(NAME audio_tests COMMAND audio_tests)

  add_executable(cli_tests tests/cli_tests.c src/cli.c)
  target_include_directories(cli_tests PRIVATE include)
  stt_enable_warnings(cli_tests)
  add_test(NAME cli_tests COMMAND cli_tests)

  add_executable(model_tests tests/model_tests.c src/model.c src/fs.c src/log.c)
  target_include_directories(model_tests PRIVATE include)
  stt_enable_warnings(model_tests)
  add_test(NAME model_tests COMMAND model_tests)

  add_executable(app_tests tests/app_tests.c src/app.c src/fs.c src/log.c)
  target_compile_definitions(app_tests PRIVATE STT_ENABLE_DOWNLOAD=0)
  target_include_directories(app_tests PRIVATE include)
  stt_enable_warnings(app_tests)
  if(WIN32)
    target_link_libraries(app_tests PRIVATE shell32 ole32)
  endif()
  add_test(NAME app_tests COMMAND app_tests)
endif()

install(TARGETS stt RUNTIME DESTINATION bin)
