#! /usr/bin/python3
import datetime
import subprocess
import re

op_list = ["conv", "gemm"]
num_tiles = [1, 32, 64, 96, 128, 160, 192, 224, 256]
width_list = [32, 45, 64, 90, 128, 181, 256, 362, 512]
num_layer_list = [1, 2, 4, 8, 16, 32, 64, 128, 256]
cm_list = [32, 45, 64, 90, 128, 181, 256, 362, 512]
output_log = "abalation.log"
output_csv = "abalation.csv"
dir_name = "abalation"

def run_abalation (op, batch, num_tile, width, num_layer, cm):
    cfg_name = op + "_W" + str(width) + "_CM" + str(cm) + "_L" + str(num_layer) + ".cfg"
    if op == "conv":
        with open (dir_name + "/" + cfg_name, "w") as f:
            f.write ("[net]\n")
            f.write ("height=" + str(width) + "\n")
            f.write ("width=" + str(width) + "\n")
            f.write ("channels=" + str(cm) + "\n\n")
            for l in range(num_layer):
                f.write ("[convolutional]" + "\n")
                f.write ("filters=" + str(cm) + "\n")
                f.write ("size=3" + "\n")
                f.write ("stride=1" + "\n")
                f.write ("pad=1" + "\n")
                f.write ("activation=linear" + "\n\n")
    else:
        with open (dir_name + "/" + cfg_name, "w") as f:
            f.write ("[net]\n")
            f.write ("height=1\n")
            f.write ("M=" + str(cm) + "\n")
            f.write ("width=" + str(cm) + "\n")
            for l in range(num_layer):
                f.write ("[matmul]" + "\n")
                f.write ("M=" + str(cm) + "\n")
                f.write ("activation=linear" + "\n\n")

    cmd = "./main " + str(batch) + " " + str(num_tile) + " 100 64 " + op + " " + str(num_layer) + " " + str(width) + " " + str(cm) 
    print (cmd + "\n")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result_str = result.stdout.decode('utf-8')
    time = re.findall(r"Time taken: (\d+\.\d+)", result_str)
    if len(time) == 0:
        print ("Error: cannot find time in the result")
        return
    time = time[0]
    print(result_str)
    with open(output_log, "a") as f:
        f.write(cmd)
        f.write(result_str)
        f.write("\n")
    with open(output_csv, "a") as f:
        f.write(op + "," + str(batch) + "," + str(num_tile) + "," + str(width) + "," + str(num_layer) + "," + str(cm) + ",")
        f.write(time + "\n")

if __name__ == "__main__":
    cmd = "make -j8"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print ("Abalation test started at " + str(datetime.datetime.now()) + "\n")
    with open(output_log, "a") as f:
        f.write ("Abalation test started at " + str(datetime.datetime.now()) + "\n")
        f.write(cmd + "\n")
        f.write(result.stdout.decode('utf-8'))
        f.write("\n")
    with open(output_csv, "a") as f:
        f.write("op,batch,num_tile,width,num_layer,cm,time\n")

    num_abalation = len(op_list) * (len(num_tiles) + len(width_list) + len(num_layer_list) + len(cm_list))
    abal_idx = 0
    for op in op_list:
        for num_tile in num_tiles:
            width = 128
            num_layer = 32
            cm = 128
            print ("Abalation test " + str(abal_idx) + "/" + str(num_abalation) + " started at " + str(datetime.datetime.now()) + "\n")
            with open(output_log, "a") as f:
                f.write ("Abalation test " + str(abal_idx) + "/" + str(num_abalation) + " started at " + str(datetime.datetime.now()) + "\n")
            abal_idx += 1
            run_abalation (op, 1, num_tile, width, num_layer, cm)
        for width in width_list:
            num_tile = 128
            num_layer = 32
            cm = 128
            print ("Abalation test " + str(abal_idx) + "/" + str(num_abalation) + " started at " + str(datetime.datetime.now()) + "\n")
            with open(output_log, "a") as f:
                f.write ("Abalation test " + str(abal_idx) + "/" + str(num_abalation) + " started at " + str(datetime.datetime.now()) + "\n")
            abal_idx += 1
            run_abalation (op, 1, num_tile, width, num_layer, cm)
        for num_layer in num_layer_list:
            num_tile = 128
            width = 128
            cm = 128
            print ("Abalation test " + str(abal_idx) + "/" + str(num_abalation) + " started at " + str(datetime.datetime.now()) + "\n")
            with open(output_log, "a") as f:
                f.write ("Abalation test " + str(abal_idx) + "/" + str(num_abalation) + " started at " + str(datetime.datetime.now()) + "\n")
            abal_idx += 1
            run_abalation (op, 1, num_tile, width, num_layer, cm)
        for cm in cm_list:
            num_tile = 128
            width = 128
            num_layer = 32
            print ("Abalation test " + str(abal_idx) + "/" + str(num_abalation) + " started at " + str(datetime.datetime.now()) + "\n")
            with open(output_log, "a") as f:
                f.write ("Abalation test " + str(abal_idx) + "/" + str(num_abalation) + " started at " + str(datetime.datetime.now()) + "\n")
            abal_idx += 1
            run_abalation (op, 1, num_tile, width, num_layer, cm)

    print ("Abalation test ended at " + str(datetime.datetime.now()) + "\n")
    with open(output_log, "a") as f:
        f.write ("Abalation test ended at " + str(datetime.datetime.now()) + "\n")