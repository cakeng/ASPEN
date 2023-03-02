#ifndef _APU_H_
#define _APU_H_

#include "aspen.h"
#include "nasm.h"

void init_aspen_dnn(aspen_dnn_t *dnn, unsigned int num_layers);
void init_aspen_layer(aspen_layer_t *layer, unsigned int layer_num, aspen_dnn_t *dnn);

#endif /* _APU_H_ */