#! /usr/bin/python3

import subprocess
dnn_list = ["resnet50", "vgg16", "yolov3", "bert_base"]
batch_list = [1, 2, 4, 8, 16, 32]
num_tiles = [20, 50, 100, 200]
output_log = "nasm_gen.log"

for dnn in dnn_list:
    for batch in batch_list:
        for num_tile in num_tiles:
            cmd = "./nasm_gen_linux_x86" + " " + dnn + " " + str(batch) + " " + str(num_tile) + " 50 32"
            print (cmd)
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode('utf-8'))
            with open(output_log, "a") as f:
                f.write(cmd)
                f.write(result.stdout.decode('utf-8'))
                f.write("\n")