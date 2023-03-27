#ifndef _UTIL_H_
#define _UTIL_H_

#include "aspen.h"
#include "nasm.h"

void *aspen_calloc (size_t num, size_t size);
void *aspen_malloc (size_t num, size_t size);
void aspen_free (void *ptr);
void *aspen_gpu_calloc (size_t num, size_t size, int gpu_num);
void *aspen_gpu_malloc (size_t num, size_t size, int gpu_num);
void aspen_gpu_free (void *ptr);
unsigned int get_smallest_dividable (unsigned int num, unsigned int divider);

#endif /* _UTIL_H_ */