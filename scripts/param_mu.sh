EDGE_CREDIT="nxc@192.168.1.25"
PIPELINED=1
DIRNAME="multiuser"
TARGET_DNN_DIR="data/vgg16_base.aspen"
# TARGET_DNN_DIR="data/resnet50_base.aspen"
# TARGET_DNN_DIR="data/bert_base_base.aspen"
# TARGET_DNN_DIR="data/yolov3_base.aspen"
TARGET_NASM_DIR="data/vgg16_B1_T100.nasm"
# TARGET_NASM_DIR="data/resnet50_B1_T100.nasm"
# TARGET_NASM_DIR="data/bert_base_B1_T100.nasm"
# TARGET_NASM_DIR="data/yolov3_B1_T100.nasm"
TARGET_INPUT="data/batched_input_128.bin"
PREFIX="vgg16_B1"
server_ip="147.46.130.51"
# server_ip="127.0.0.1"
server_port=8081
# SCHEDULE_POLICY="local"
# SCHEDULE_POLICY="dynamic"
# SCHEDULE_POLICY="sequential"
# SCHEDULE_POLICY="conventional"
# SCHEDULE_POLICY="conventional+pipeline"
SCHEDULE_POLICY="spinn"
# SCHEDULE_POLICY="spinn+pipeline"
SCHED_SEQUENTIAL_IDX=1
DSE_NUM=16
OUTPUT_ORDER="cnn"
INFERENCE_REPEAT_NUM=1
NUM_EDGE_DEVICES=1
EDGE_DSE_NUM=6