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
  json-c
  libcurl
  libonnxruntime
  fftw3f
  samplerate
  cuda
  cublas
  cufft
)

file(GLOB STT_SOURCES CONFIGURE_DEPENDS src/*.c)
add_executable(stt ${STT_SOURCES})

target_include_directories(stt PRIVATE include)
target_include_directories(stt SYSTEM PRIVATE ${STT_DEPS_INCLUDE_DIRS})
target_compile_options(stt PRIVATE -Wall -Wextra -Wpedantic ${STT_DEPS_CFLAGS_OTHER})
target_link_directories(stt PRIVATE ${STT_DEPS_LIBRARY_DIRS})
target_link_libraries(stt PRIVATE ${STT_DEPS_LIBRARIES} m pthread)

install(TARGETS stt RUNTIME DESTINATION bin)
