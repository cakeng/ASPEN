# Code from https://pytorch.org/hub/pytorch_vision_resnet/
import torch
import sys
from PIL import Image
from torchvision import transforms

def dump_tensor (path, tensor):
    np_arr = tensor.detach().numpy()
    np_arr.astype('float32').tofile(path)

# Load the images
dog_image = Image.open("../../files/dog.jpg")
cat_image = Image.open("../../files/cat.jpg")
penguin_image = Image.open("../../files/penguin.jpg")
bunny_image = Image.open("../../files/bunny.jpg")

# Preprocess the images into tensors
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dog_tensor = preprocess(dog_image)
cat_tensor = preprocess(cat_image)
penguin_tensor = preprocess(penguin_image)
bunny_tensor = preprocess(bunny_image)
input_batch = dog_tensor.unsqueeze(0)
input_batch =  torch.cat((input_batch, cat_tensor.unsqueeze(0)), 0)
input_batch =  torch.cat((input_batch, penguin_tensor.unsqueeze(0)), 0)
input_batch =  torch.cat((input_batch, bunny_tensor.unsqueeze(0)), 0)

# Save the tensors
dump_tensor("dog.tensor", dog_tensor)
dump_tensor("cat.tensor", cat_tensor)
# dump_tensor("penguin.tensor", penguin_tensor)
# dump_tensor("bunny.tensor", bunny_tensor)
dump_tensor("batched_input.tensor", input_batch)

