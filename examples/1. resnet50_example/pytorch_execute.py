# Code from https://pytorch.org/hub/pytorch_vision_resnet/
import torch
import time
import os
import sys
import numpy as np
import torchvision

num_threads = os.cpu_count()
torch.set_num_threads(num_threads)
batch_size = 1
model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
model.eval()

def read_tensor (path):
    np_arr = np.fromfile(path, dtype='float32')
    tensor = torch.from_numpy(np_arr)
    tensor = tensor.reshape((batch_size, 3, 224, 224))
    return tensor

if len(sys.argv) != 3:
    print ("Usage: python3 pytorch_resnet50.py <input_tensor_dir> <number_of_iterations>")
    exit(1)
else:
    input_tensor = read_tensor(sys.argv[1])
    number_of_iterations = int(sys.argv[2])

time_start = time.time()
for i in range (number_of_iterations):
    with torch.no_grad():
        output = model(input_tensor)
time_end = time.time()
print("Average time taken (" + str(number_of_iterations) + " runs) : %3.6f" % ((time_end - time_start)/number_of_iterations), "s")

for b in range(batch_size):
    probabilities = torch.nn.functional.softmax(output[b], dim=0)
    print ("Batch ", str(b+1), " results:")
    with open("../../files/imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    for i in range(top5_prob.size(0)):
        print("\t" + str(i+1) + ": " +categories[top5_catid[i]], "- %2.2f%%" % (top5_prob[i]*100))