#! /usr/bin/python3

import subprocess
output_log = "nasm_gen.log"
# dnn_list = []
# batch_list = range (1, 13)

# for dnn in dnn_list:
#     for batch in batch_list:
#         cmd = "./main" + " " + dnn + " " + str(batch) 
#         print (cmd)
#         result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         print(result.stdout.decode('utf-8'))
#         with open(output_log, "a") as f:
#             f.write(cmd)
#             f.write(result.stdout.decode('utf-8'))
#             f.write("\n")

# dnn_list = ["yolov3", "vgg16"]
# batch_list = range (17, 33)

# for dnn in dnn_list:
#     for batch in batch_list:
#         cmd = "./main" + " " + dnn + " " + str(batch)
#         print (cmd)
#         result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#         print(result.stdout.decode('utf-8'))
#         with open(output_log, "a") as f:
#             f.write(cmd)
#             f.write(result.stdout.decode('utf-8'))
#             f.write("\n")


# dnn_list = ["bert_base", "bert_large", "gpt2_124M"]

# batch_list = range (5, 17) 
# seq_list = range (1, 32) 

# for dnn in dnn_list:
#     for batch in batch_list:
#         for seq in seq_list:
#             cmd = "./main" + " " + dnn + " " + str(batch) + " " + str(seq)
#             print (cmd)
#             result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             print(result.stdout.decode('utf-8'))
#             with open(output_log, "a") as f:
#                 f.write(cmd)
#                 f.write(result.stdout.decode('utf-8'))
#                 f.write("\n")

# batch_list = [10, 12, 14, 16]
# seq_list = range (32, 64, 2)

# for dnn in dnn_list:
#     for batch in batch_list:
#         for seq in seq_list:
#             cmd = "./main" + " " + dnn + " " + str(batch) + " " + str(seq)
#             print (cmd)
#             result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#             print(result.stdout.decode('utf-8'))
#             with open(output_log, "a") as f:
#                 f.write(cmd)
#                 f.write(result.stdout.decode('utf-8'))
#                 f.write("\n")

dnn_list = ["bert_large", "gpt2_124M"]

batch_list = [8, 10, 12, 14, 16]
seq_list = range (64, 256, 4)

for dnn in dnn_list:
    for batch in batch_list:
        for seq in seq_list:
            cmd = "./main" + " " + dnn + " " + str(batch) + " " + str(seq)
            print (cmd)
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode('utf-8'))
            with open(output_log, "a") as f:
                f.write(cmd)
                f.write(result.stdout.decode('utf-8'))
                f.write("\n")

dnn_list = ["bert_base", "bert_large", "gpt2_124M"]

batch_list = [4, 6, 8]
seq_list = range (256, 513, 8) 

for dnn in dnn_list:
    for batch in batch_list:
        for seq in seq_list:
            cmd = "./main" + " " + dnn + " " + str(batch) + " " + str(seq)
            print (cmd)
            result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(result.stdout.decode('utf-8'))
            with open(output_log, "a") as f:
                f.write(cmd)
                f.write(result.stdout.decode('utf-8'))
                f.write("\n")
        
