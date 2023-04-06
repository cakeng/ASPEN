#include "kernels.h"

#define __L1_CACHE 32768
#define __L2_CACHE 1024*1024

void prepare_input (ninst_t *ninst, LAYER_TYPE op, void *spad)
{
    if (op == CONV_LAYER)
    {
        
    }
}



void tiled_conv2d (ninst_t *ninst)
{

}
void tiled_maxpool2d (ninst_t *ninst)
{

}
void tiled_avgpool2d (ninst_t *ninst)
{

}
void tiled_fully_connected (ninst_t *ninst)
{

}
void tiled_residual (ninst_t *ninst)
{

}
void tiled_softmax (ninst_t *ninst)
{

}