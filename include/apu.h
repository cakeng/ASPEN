#ifndef _APU_H_
#define _APU_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "aspen.h"
#include "nasm.h"

aspen_dnn_t *init_aspen_dnn(unsigned int num_layers, char* name);
void init_aspen_layer(aspen_layer_t *layer, unsigned int layer_num, aspen_dnn_t *dnn);

#endif /* _APU_H_ */