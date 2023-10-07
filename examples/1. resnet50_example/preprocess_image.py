# Code from https://pytorch.org/hub/pytorch_vision_resnet/
import torch
import sys
from PIL import Image
from torchvision import transforms

def dump_tensor (path, tensor):
    np_arr = tensor.detach().numpy()
    np_arr.astype('float32').tofile(path)

if len(sys.argv) != 3:
    print ("Usage: python3 preprocess_image.py <input_img_dir> <output_tensor_dir>")
    exit(1)
else:
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

# Preprocess the image
batch_size = int(1)
input_image = Image.open(input_dir)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(input_image)
input_batch = input_tensor.unsqueeze(0) 
for i in range (batch_size - 1):
    input_batch =  torch.cat((input_batch, input_tensor.unsqueeze(0)), 0)
dump_tensor(output_dir, input_batch)
