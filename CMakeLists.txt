cmake_minimum_required(VERSION 3.22)
project(stt C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(STT_DEPS REQUIRED
  libpulse-simple
  x11
  xtst
  libonnxruntime
  fftw3f
  cuda
)

file(GLOB STT_SOURCES CONFIGURE_DEPENDS src/*.c)
add_executable(stt ${STT_SOURCES})

target_include_directories(stt PRIVATE include)
target_include_directories(stt SYSTEM PRIVATE ${STT_DEPS_INCLUDE_DIRS})
target_compile_options(stt PRIVATE -Wall -Wextra -Wpedantic ${STT_DEPS_CFLAGS_OTHER})
target_link_directories(stt PRIVATE ${STT_DEPS_LIBRARY_DIRS})
target_link_libraries(stt PRIVATE ${STT_DEPS_LIBRARIES} m pthread)

include(CTest)
if(BUILD_TESTING)
  add_executable(hotkey_tests tests/hotkey_tests.c src/log.c)
  target_include_directories(hotkey_tests PRIVATE include)
  target_compile_options(hotkey_tests PRIVATE -Wall -Wextra -Wpedantic)
  add_test(NAME hotkey_tests COMMAND hotkey_tests)

  add_executable(audio_tests tests/audio_tests.c src/audio.c src/log.c)
  target_compile_definitions(audio_tests PRIVATE STT_TESTING)
  target_include_directories(audio_tests PRIVATE include)
  target_include_directories(audio_tests SYSTEM PRIVATE ${STT_DEPS_INCLUDE_DIRS})
  target_compile_options(audio_tests PRIVATE -Wall -Wextra -Wpedantic ${STT_DEPS_CFLAGS_OTHER})
  target_link_directories(audio_tests PRIVATE ${STT_DEPS_LIBRARY_DIRS})
  target_link_libraries(audio_tests PRIVATE ${STT_DEPS_LIBRARIES} pthread)
  add_test(NAME audio_tests COMMAND audio_tests)
endif()

install(TARGETS stt RUNTIME DESTINATION bin)
