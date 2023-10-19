EDGE_CREDIT="nxc@127.0.0.1"
PIPELINED=1
DIRNAME="test"
# TARGET_DNN_DIR="data/vgg16_base.aspen"
TARGET_DNN_DIR="data/resnet50_base.aspen"
# TARGET_DNN_DIR="data/bert_base_base.aspen"
# TARGET_DNN_DIR="data/yolov3_base.aspen"
# TARGET_NASM_DIR="data/vgg16_B1_T100.nasm"
TARGET_NASM_DIR="data/resnet50_B1_T100.nasm"
# TARGET_NASM_DIR="data/bert_base_B1_T100.nasm"
# TARGET_NASM_DIR="data/yolov3_B1_T100.nasm"
TARGET_INPUT="data/batched_input_128.bin"
# PREFIX="vgg16"
PREFIX="resnet50"
# server_ip="127.0.0.1"
server_ip="147.46.66.99"
server_port=62000
# SCHEDULE_POLICY="local"
# SCHEDULE_POLICY="partial"
# SCHEDULE_POLICY="random"
SCHEDULE_POLICY="dynamic"
# SCHEDULE_POLICY="sequential"
# SCHEDULE_POLICY="conventional"
# SCHEDULE_POLICY="conventional+pipeline"
# SCHEDULE_POLICY="spinn"
# SCHEDULE_POLICY="spinn+pipeline"
SCHED_SEQUENTIAL_IDX=1
SCHED_PARTIAL_RATIO=0
DSE_NUM=2
OUTPUT_ORDER="cnn"
# OUTPUT_ORDER="transformer"
INFERENCE_REPEAT_NUM=10
NUM_EDGE_DEVICES=1
EDGE_DSE_NUM=2
