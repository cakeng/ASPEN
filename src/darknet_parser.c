#include "input_parser.h"

//// Darknet CFG parser code taken from the Darknet repo. ////
//// https://github.com/pjreddie/darknet ////

typedef struct node{
    void *val;
    struct node *next;
    struct node *prev;
} node;

typedef struct list{
    int size;
    node *front;
    node *back;
} list;

typedef struct{
    char *key;
    char *val;
    int used;
} kvp;

typedef struct{
    int classes;
    char **names;
} metadata;

typedef struct{
    char *type;
    list *options;
} section;

void error(const char *s)
{
    perror(s);
    assert(0);
    exit(-1);
}

void malloc_error()
{
    fprintf(stderr, "Malloc error\n");
    exit(-1);
}

void file_error(char *s)
{
    fprintf(stderr, "Couldn't open file: %s\n", s);
    exit(0);
}

list *make_list()
{
	list *l = malloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
}

void *list_pop(list *l){
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;
    
    return val;
}

void list_remove (list *l, node *n)
{
    if(!n->prev) l->front = n->next;
    else n->prev->next = n->next;
    if(!n->next) l->back = n->prev;
    else n->next->prev = n->prev;
    --l->size;
    free(n);
}

void list_insert(list *l, void *val)
{
	node *new = malloc(sizeof(node));
	new->val = val;
	new->next = 0;

	if(!l->back){
		l->front = new;
		new->prev = 0;
	}else{
		l->back->next = new;
		new->prev = l->back;
	}
	l->back = new;
	++l->size;
}

void free_node(node *n)
{
	node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}

void free_list(list *l)
{
	free_node(l->front);
	free(l);
}

void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}

void free_section(section *s)
{
    free(s->type);
    node *n = s->options->front;
    while(n){
        kvp *pair = (kvp *)n->val;
        free(pair->key);
        free(pair);
        node *next = n->next;
        free(n);
        n = next;
    }
    free(s->options);
    free(s);
}

void **list_to_array(list *l)
{
    void **a = calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}

void strip(char *s)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==' '||c=='\t'||c=='\n') ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}

void strip_char(char *s, char bad)
{
    size_t i;
    size_t len = strlen(s);
    size_t offset = 0;
    for(i = 0; i < len; ++i){
        char c = s[i];
        if(c==bad) ++offset;
        else s[i-offset] = c;
    }
    s[len-offset] = '\0';
}

char *fgetl(FILE *fp)
{
    if(feof(fp)) return 0;
    size_t size = 512;
    char *line = malloc(size*sizeof(char));
    if(!fgets(line, size, fp)){
        free(line);
        return 0;
    }

    size_t curr = strlen(line);

    while((line[curr-1] != '\n') && !feof(fp)){
        if(curr == size-1){
            size *= 2;
            line = realloc(line, size*sizeof(char));
            if(!line) {
                printf("%ld\n", size);
                malloc_error ();
            }
        }
        size_t readsize = size-curr;
        if(readsize > INT_MAX) readsize = INT_MAX-1;
        char* tmp = fgets(&line[curr], readsize, fp);
        if(!tmp)
        {
            free(line);
            return 0;
        }
        curr = strlen(line);
    }
    if(line[curr-1] == '\n') line[curr-1] = '\0';

    return line;
}

list *get_paths(char *filename)
{
    char *path;
    FILE *file = fopen(filename, "r");
    if(!file) file_error(filename);
    list *lines = make_list();
    while((path=fgetl(file))){
        list_insert(lines, path);
    }
    fclose(file);
    return lines;
}

void option_insert(list *l, char *key, char *val)
{
    kvp *p = malloc(sizeof(kvp));
    p->key = key;
    p->val = val;
    p->used = 0;
    list_insert(l, p);
}

void option_unused(list *l)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(!p->used){
            FPRT(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
        }
        n = n->next;
    }
}

char *option_find(list *l, char *key)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(strcmp(p->key, key) == 0){
            p->used = 1;
            return p->val;
        }
        n = n->next;
    }
    return 0;
}
char *option_find_str(list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    return def;
}

int option_find_int(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    FPRT(stderr, "%s: Using default '%d'\n", key, def);
    return def;
}

int option_find_int_quiet(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;
}

float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;
}

float option_find_float(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    FPRT(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
}

int read_option(char *s, list *options)
{
    size_t i;
    size_t len = strlen(s);
    char *val = 0;
    for(i = 0; i < len; ++i){
        if(s[i] == '='){
            s[i] = '\0';
            val = s+i+1;
            break;
        }
    }
    if(i == len-1) return 0;
    char *key = s;
    option_insert(options, key, val);
    return 1;
}

char **get_labels(char *filename)
{
    list *plist = get_paths(filename);
    char **labels = (char **)list_to_array(plist);
    free_list(plist);
    return labels;
}

list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == NULL)
    {
        FPRT (stderr, "Couldn't open file: %s\n", filename);
        return NULL;
    }
    char *line;
    int nu = 0;
    list *options = make_list();
    section *current = 0;
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '[':
                current = malloc(sizeof(section));
                list_insert(options, current);
                current->options = make_list();
                current->type = line;
                break;
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, current->options)){
                    FPRT (stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

metadata get_metadata(char *file)
{
    metadata m = {0};
    list *options = read_data_cfg(file);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", 0);
    if(!name_list) {
        FPRT(stderr, "No names or labels found\n");
    } else {
        m.names = get_labels(name_list);
    }
    m.classes = option_find_int(options, "classes", 2);
    free_list(options);
    return m;
}

void parse_input_layer_options (list *options, aspen_layer_t *layer)
{
    // net->batch = option_find_int(options, "batch",1);
    // int subdivs = option_find_int(options, "subdivisions",1);
    // net->batch /= subdivs;
    // net->batch *= net->time_steps;
    // net->subdivisions = subdivs;

    // net->h = option_find_int_quiet(options, "height",0);
    // net->w = option_find_int_quiet(options, "width",0);
    // net->c = option_find_int_quiet(options, "channels",0);
    // net->inputs = option_find_int_quiet(options, "inputs", net->h * net->w * net->c);
    // if(!net->inputs && !(net->h && net->w && net->c)) error("No input parameters supplied");
}

int is_network(section *s)
{
    return (strcmp(s->type, "[net]")==0
            || strcmp(s->type, "[network]")==0);
}

LAYER_ACT get_activation(char *s)
{
    if (!s) return NO_ACTIVATION;
    if (strcmp(s, "logistic")==0) return SIGMOID;
    if (strcmp(s, "linear")==0) return NO_ACTIVATION;
    if (strcmp(s, "leaky")==0) return LEAKY_RELU;
    if (strcmp(s, "relu")==0) return RELU;
    if (strcmp(s, "elu")==0) return ELU;
    if (strcmp(s, "selu")==0) return SELU;
    if (strcmp(s, "tanh")==0) return TANH;
    if (strcmp(s, "gelu")==0) return GELU;
    fprintf(stderr, "Couldn't find activation function %s, going with ReLU\n", s);
    return RELU;
}

LAYER_TYPE string_to_layer_type(char * type)
{
    if (strcmp(type, "[net]")==0
            || strcmp(type, "[network]")==0) return INPUT_LAYER;
    if (strcmp(type, "[conv]")==0
            || strcmp(type, "[convolutional]")==0) return CONV_LAYER;
    if (strcmp(type, "[shortcut]")==0) return RESIDUAL_LAYER;
    if (strcmp(type, "[batchnorm]")==0) return BATCHNORM_LAYER;
    if (strcmp(type, "[yolo]")==0) return YOLO_LAYER;
    if (strcmp(type, "[activation]")==0) return ACTIVATION_LAYER;
    if (strcmp(type, "[max]")==0
            || strcmp(type, "[maxpool]")==0) return MAXPOOL_LAYER;
    if (strcmp(type, "[avg]")==0
            || strcmp(type, "[avgpool]")==0) return AVGPOOL_LAYER;
    if (strcmp(type, "[route]")==0) return ROUTE_LAYER;
    if (strcmp(type, "[conn]")==0
            || strcmp(type, "[connected]")==0) return FC_LAYER;
    if (strcmp(type, "[soft]")==0
            || strcmp(type, "[softmax]")==0) return SOFTMAX_LAYER;
    if (strcmp(type, "[yolo]")==0) return YOLO_LAYER;
    if (strcmp(type, "[append]")==0) return APPEND_LAYER;
    if (strcmp(type, "[matmul]")==0) return MATMUL_LAYER;
    if (strcmp(type, "[attention_k]")==0) return K_ATTENTION_LAYER;
    if (strcmp(type, "[attention_v]")==0) return V_ATTENTION_LAYER;
    if (strcmp(type, "[layernorm]")==0) return LAYERNORM_LAYER;
    return NO_LAYER_TYPE;
}

void remove_unsupported_sections (list *sections)
{
    node *n = sections->front;
    while (n) 
    {
        section *s = (section *) n->val;
        node *next = n->next;
        if (string_to_layer_type (s->type) == NO_LAYER_TYPE) 
        {
            FPRT (stderr, "Unsupported section type: %s\n", s->type);
            free_section(s);
            list_remove (sections, n);
        }
        n = next;
    }
}
// Change to add a new layer type
void parse_section (section *s, aspen_layer_t *layer)
{
    layer->type = string_to_layer_type (s->type);
    list *options = s->options;
    if (layer->type == NO_LAYER_TYPE) FPRT (stderr, "Unknown layer type: %s", s->type);
    layer->params [IN_W] = option_find_int_quiet (options, "width", 0);
    layer->params [IN_H] = option_find_int_quiet (options, "height", 0);
    layer->params [IN_C] = option_find_int_quiet (options, "channels", 0);
    layer->params [WEIGHT_W] = option_find_int_quiet (options, "size", 0);
    layer->params [WEIGHT_H] = option_find_int_quiet (options, "size", 0);
    layer->params [OUT_C] = option_find_int_quiet (options, "filters", 0);
    if (layer->params [OUT_C] == 0) layer->params [OUT_C] = option_find_int_quiet (options, "output", 0);
    layer->params [STRIDE] = option_find_int_quiet (options, "stride", 0);
    layer->params [PADDING] = option_find_int_quiet (options, "pad", 0);
    layer->params [DILATION] = option_find_int_quiet (options, "dilation", 0);
    layer->params [MAT_M] = option_find_int_quiet (options, "M", 0);
    layer->params [MAT_N] = option_find_int_quiet (options, "N", 0);
    layer->params [MAT_K] = option_find_int_quiet (options, "K", 0);
    layer->params [NUM_HIDDEN] = option_find_int_quiet (options, "n_embd", 0);
    layer->params [NUM_HEAD] = option_find_int_quiet (options, "n_head", 0);
    layer->params [MASKED] = option_find_int_quiet (options, "masked", 0);
    char *activation_s = option_find_str(options, "activation", NULL);
    layer->activation = get_activation(activation_s);
    layer->parent_layers [PARENT_0] = layer + option_find_int_quiet (options, "parent", 0);
    if (layer->type != INPUT_LAYER && layer->parent_layers [PARENT_0] == layer)
        layer->parent_layers [PARENT_0] = layer - 1;
    else if (layer->type == INPUT_LAYER)
        layer->parent_layers [PARENT_0] = NULL;
    if (option_find_int_quiet (options, "from", 0))
    {
        layer->parent_layers [PARENT_1] = layer + option_find_int_quiet (options, "from", 0);
    }
    if (layer->type == YOLO_LAYER)
    {
        layer->params [OUT_C] = option_find_int_quiet (options, "classes", 0) + 5;
        int *masks = NULL;
        float *anchors = NULL;
        char *a = option_find_str(options, "mask", 0);
        if(a)
        {
            int len = strlen(a);
            int n = 1;
            int i;
            for(i = 0; i < len; ++i)
            {
                if (a[i] == ',') 
                    n++;
            }
            masks = calloc(n, sizeof(int));
            for(i = 0; i < n; ++i)
            {
                masks[i] = atoi(a);
                a = strchr(a, ',') + 1;
            }
        }
        else
        {
            FPRT (stderr, "YOLO layer has no mask.\n");
        }
        a = option_find_str(options, "anchors", 0);
        if(a)
        {
            int len = strlen(a);
            int n = 1;
            int i;
            for(i = 0; i < len; ++i)
            {
                if (a[i] == ',') 
                    n++;
            }
            anchors = calloc(n, sizeof(float));
            for(i = 0; i < n; ++i)
            {
                anchors[i] = atof(a);
                a = strchr(a, ',') + 1;
            }
        }
        else
        {
            FPRT (stderr, "YOLO layer has no anchors.\n");
        }
        LAYER_PARAMS dim_order[] = {OUT_C};
        unsigned int params[NUM_PARAM_ELEMENTS];
        for (int i = 0; i < NUM_PARAM_ELEMENTS; i++)
            params[i] = 6;
        layer->tensors [ANCHOR_TENSOR] = init_aspen_tensor (params, dim_order, 1, layer->dnn->element_size);
        calloc_aspen_tensor (layer->tensors [ANCHOR_TENSOR]);
        for (int i = 0; i < 3; i++)
        {
            ((float*)layer->tensors[ANCHOR_TENSOR]->data)[i*2] = anchors[masks[i]*2];
            ((float*)layer->tensors[ANCHOR_TENSOR]->data)[i*2 + 1] = anchors[masks[i]*2 + 1];
        }
        // print_tensor_info (layer->tensors [ANCHOR_TENSOR], 1);
        if (masks)
            free (masks);
        if (anchors)
            free (anchors);
    }
}

aspen_dnn_t *parse_darknet_cfg (char *filename)
{
    PRT ("Parsing Darknet CFG file: %s\n", filename);
    list *sections = read_data_cfg(filename);
    node *n = sections->front;
    if(!n) error ("Config file has no sections");
    remove_unsupported_sections (sections);
    PRT ("Config file has %d sections \n", sections->size);
    aspen_dnn_t *dnn = init_aspen_dnn (sections->size, filename);

    section *s = (section *)n->val;
    if(!is_network(s)) error("First section must be [net] or [network]");
    for (int i = 0; i < sections->size; i++)
    {
        s = (section *)n->val;
        parse_section (s, dnn->layers + i);
        free_section(s);
        n = n->next;
    }
    free_list(sections);
    return dnn;
}