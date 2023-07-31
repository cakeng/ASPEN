PIPELINED=1
DIRNAME="temp"
TARGET_DNN_DIR="data/vgg16_base.aspen"
TARGET_NASM_DIR="data/vgg16_B2_T100.nasm"
TARGET_INPUT="data/batched_input_128.bin"
PREFIX="vgg16_B1"
server_ip="127.0.0.1"
server_port=53488
SCHEDULE_POLICY="dynamic"
SCHED_SEQUENTIAL_IDX=1
DSE_NUM=16
OUTPUT_ORDER="cnn"
INFERENCE_REPEAT_NUM=1