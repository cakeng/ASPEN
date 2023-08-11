PIPELINED=1
DIRNAME="temp"
TARGET_DNN_DIR="data/vgg16_base.aspen"
# TARGET_DNN_DIR="data/resnet50_base.aspen"
TARGET_NASM_DIR="data/vgg16_B2_T100.nasm"
# TARGET_NASM_DIR="data/resnet50_B32_T100.nasm"
TARGET_INPUT="data/batched_input_128.bin"
PREFIX="vgg16_B2"
server_ip="147.46.130.51"
server_port=8081
# SCHEDULE_POLICY="local"
SCHEDULE_POLICY="dynamic"
# SCHEDULE_POLICY="sequential"
# SCHEDULE_POLICY="conventional"
SCHED_SEQUENTIAL_IDX=1
<<<<<<< HEAD
DSE_NUM=1
=======
DSE_NUM=16
>>>>>>> 258de8947df9b6cb47d8680591f75f80c9ef3965
OUTPUT_ORDER="cnn"
INFERENCE_REPEAT_NUM=5