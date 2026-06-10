#ifndef STT_FS_H
#define STT_FS_H

#include <stddef.h>

char *stt_expand_home(const char *path);
char *stt_path_join(const char *a, const char *b);
int stt_mkdir_p(const char *path);
int stt_file_exists(const char *path);
char *stt_read_file(const char *path, size_t *len_out);

#endif
