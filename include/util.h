#ifndef _UTIL_H_
#define _UTIL_H_

#include "aspen.h"
#include "nasm.h"

extern char *layer_type_str [NUM_LAYER_ELEMENTS];
extern char *param_type_str[NUM_PARAM_ELEMENTS];
extern char *tensor_type_str[NUM_TENSOR_ELEMENTS];
extern char *parent_type_str[NUM_PARENT_ELEMENTS];
extern char *activation_type_str [NUM_ACTIVATION_ELEMENTS];
extern char *nist_op_str [NUM_NIST_OP_ELEMENTS];

void *aspen_calloc (size_t num, size_t size);
void *aspen_malloc (size_t num, size_t size);
void aspen_free (void *ptr);
void *aspen_gpu_calloc (size_t num, size_t size, int gpu_num);
void *aspen_gpu_malloc (size_t num, size_t size, int gpu_num);
void aspen_gpu_free (void *ptr);
unsigned int get_smallest_dividable (unsigned int num, unsigned int divider);

#endif /* _UTIL_H_ */