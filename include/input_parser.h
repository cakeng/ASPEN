#ifndef _INPUT_PARSER_H_
#define _INPUT_PARSER_H_

#include "aspen.h"
#include "apu.h"
#include "nasm.h"

// Darknet parser code taken from the Darknet repo.
typedef enum {
    darknet_cfg,
} input_type;

aspen_dnn_t *parse_input (char *filename);

input_type get_input_type(char *input);

aspen_dnn_t *parse_darknet_cfg (char *filename);

void set_layer_inout_sizes (aspen_layer_t *layer);

#endif // _INPUT_PARSER_H_