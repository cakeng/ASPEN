#ifndef _UTIL_H_
#define _UTIL_H_

#include "aspen.h"
#include "nasm.h"

void *aspen_calloc (size_t num, size_t size);
void *aspen_malloc (size_t num, size_t size);
void aspen_free (void *ptr);
void *aspen_gpu_calloc (size_t num, size_t size, int gpu_num);
void *aspen_gpu_malloc (size_t num, size_t size, int gpu_num);

void aspen_host_to_gpu_memcpy (void *dst, void *src, size_t num, int gpu_num);
void aspen_gpu_to_host_memcpy (void *dst, void *src, size_t num, int gpu_num);
void aspen_host_to_gpu_async_memcpy (void *dst, void *src, size_t num, int gpu_num, int stream_num);
void aspen_gpu_to_host_async_memcpy (void *dst, void *src, size_t num, int gpu_num, int stream_num);
void aspen_sync_gpu (int gpu_num);
void aspen_sync_gpu_stream (int gpu_num, int stream_num);

int aspen_get_next_stream (int gpu_num);

void aspen_gpu_free (void *ptr, int gpu_num);
unsigned int get_smallest_dividable (unsigned int num, unsigned int divider);

#endif /* _UTIL_H_ */