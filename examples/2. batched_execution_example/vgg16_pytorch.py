# Code from https://pytorch.org/vision/main/_modules/torchvision/models/vgg.html
import torch
import torch.nn as nn
import time
import os
import numpy as np
from typing import Union, List, Dict, Any, cast
from torch.hub import load_state_dict_from_url

# PyTorch implementation of VGG-16 from TorchVision.
model_urls = {
    "vgg16": "https://download.pytorch.org/models/vgg16-397923af.pth",
}

cfgs: Dict[str, List[Union[str, int]]] = {
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
}

class VGG(nn.Module):
    layer_num = 0
    def __init__(
        self, features: nn.Module, num_classes: int = 1000, init_weights: bool = True, dropout: float = 0.5
    ) -> None:
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def make_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs["init_weights"] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model

def vgg16(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg16", "D", False, pretrained, progress, **kwargs)

# Function to dump tensors into a file with the following format:

# ASPEN_DATA
# LAYER:<layer_number>
# TENSOR_TYPE:<tensor_type>
# DATA_SIZE:<data_size>
# DATA_START:
# <tensor_data>
# DATA_END
# LAYER_END
# LAYER:<layer_number>
# ...
# LAYER_END

# ASPEN requires the above format to read the model tensors from a file.

def dump_tensor (path, tensor, tensor_info_string):
    with open(path, "a") as f:
        f.write(tensor_info_string)
    np_arr = tensor.detach().numpy()
    np_arr.astype('float32').tofile(path + ".tmp")
    data_size = os.path.getsize(path + ".tmp")
    with open(path, "a") as f:
        f.write("DATA_SIZE:" + str(data_size) + "\n")
        f.write("DATA_START:\n")
    os.system ("cat " + path + ".tmp >> " + path)
    with open(path, "a") as f:
        f.write("DATA_END\n")
    os.system ("rm " + path + ".tmp")
    with open(path, "a") as f:
        f.write("LAYER_END\n")

def dump_data():
    model = vgg16(True, False)
    model.eval()
    path = "vgg16_weight.bin"
    # print (model)
    os.system ("echo ASPEN_DATA > " + path)
    layer_idx = 1
    # Dump the weights and biases of the model.
    for layer in model.features:
        if isinstance(layer, nn.Conv2d):
            dump_tensor (path, layer.weight, "LAYER:" + str(layer_idx) + "\nTENSOR_TYPE:WEIGHT\n")
            dump_tensor (path, layer.bias, "LAYER:" + str(layer_idx) + "\nTENSOR_TYPE:BIAS\n")
            layer_idx += 1
    for layer in model.classifier:
        if isinstance(layer, nn.Linear):
            dump_tensor (path, layer.weight, "LAYER:" + str(layer_idx) + "\nTENSOR_TYPE:WEIGHT\n")
            dump_tensor (path, layer.bias, "LAYER:" + str(layer_idx) + "\nTENSOR_TYPE:BIAS\n")
            layer_idx += 1

def read_tensor (path):
    np_arr = np.fromfile(path, dtype='float32')
    tensor = torch.from_numpy(np_arr)
    tensor = tensor.reshape((3, 224, 224))
    return tensor

model = vgg16(True, False)
model.eval()

# Dump the VGG-16 model tensors into a file
dump_data ()

# Test execution of VGG-16 on PyTorch
print ("Testing VGG-16 on PyTorch")
num_threads = os.cpu_count()
torch.set_num_threads(num_threads)
batch_size = 2
number_of_iterations = 100

dog_tensor = read_tensor("dog.tensor")
cat_tensor = read_tensor("cat.tensor")
input_batch = dog_tensor.unsqueeze(0)
input_batch =  torch.cat((input_batch, cat_tensor.unsqueeze(0)), 0)

time_start = time.time()
for i in range (number_of_iterations):
    with torch.no_grad():
        output = model(input_batch)
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
