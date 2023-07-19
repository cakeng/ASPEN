./main_scheduling \
    --sock_type=0   \
    --sequential=1  \
    --dirname=temp  \
    --target_config="data/cfg/vgg16_aspen.cfg"   \
    --target_nasm_dir="data/vgg16_B1_aspen.nasm" \
    --target_bin="data/vgg16/vgg16_data.bin" \
    --target_input="data/resnet50/batched_input_64.bin" \
    --prefix="vgg16_B1" \
    --rx_ip="192.168.1.176" \
    --rx_port=3786   \
    --schedule_policy="sequential"  \
    --sched_sequential_idx=5    \
    --dse_num=16    \
    --output_order="cnn"    \
    --inference_repeat_num=10