#include "input_parser.h"

aspen_dnn_t *parse_input (char *filename)
{
    input_type type = get_input_type(filename);
    switch(type){
        case darknet_cfg:
            return parse_darknet_cfg(filename);
        default:
            FPRT (stderr, "Input file type not supported: %s\n", filename);
            return NULL;
    }
}

input_type get_input_type (char *filename)
{
    char *ext = strrchr(filename, '.');
    if (ext == NULL) return 0;
    if (strcmp(ext, ".cfg") == 0) return darknet_cfg;
    return 0;
}
