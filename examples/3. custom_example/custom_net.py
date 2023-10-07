# Code from https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

def dump_raw_tensor (path, tensor):
    np_arr = tensor.detach().numpy()
    np_arr.astype('float32').tofile(path)
    print ("Dumped tensor with shape", tensor.shape  ,"to", path)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 10

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Define a custom network
# Network specification:
#   Layer 1: A convolution layer with 32 filters of size 5x5, stride 1 and pad 2, and a ReLU activation.
#   Layer 2: A maxpool layer with size 2x2 and stride 2.
#   Layer 3: A convolution layer with 64 filters of size 3x3, stride 1 and pad 1, and a ReLU activation.
#   Layer 4: A maxpool layer with size 2x2 and stride 2.
#   Layer 5: A fully connected (linear) layer with 256 filters and a ReLU activation.
#   Layer 6: A fully connected (linear) layer with 128 filters and a ReLU activation.
#   Layer 7: A fully connected (linear) layer with 10 filters and a linear (no) activation.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, 1, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x))) # Layer 1 and 2
        x = self.pool(F.relu(self.conv2(x))) # Layer 3 and 4
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x)) # Layer 5
        x = F.relu(self.fc2(x)) # Layer 6
        x = self.fc3(x) # Layer 7
        return x

net = Net()

# Train the custom network
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
for epoch in range(4):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 999:    # print every 1000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 1000:.3f}')
            running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), "custom.pth")

# Load the custom network
net.load_state_dict(torch.load("custom.pth"))

# Test the custom network
net.eval()
dataiter = iter(testloader)
images, labels = next(dataiter)
outputs = net(images)
_, predicted = torch.max(outputs, 1)


print("Ground Truth: ", end = " ")
for j in range(batch_size):
    print(classes[labels[j]], end = " ")
print()
print("Predicted: ", end = " ")
for j in range(batch_size):
    print(classes[predicted[j]], end = " ")
print()

# Dump the input and network weights.
# ASPEN weight .bin file format:
#   ASPEN_DATA
#   LAYER:<layer_idx>
#   TENSOR_TYPE:<tensor_type>
#   DATA_SIZE:<data_size>
#   DATA_START:
#   <data>
#   DATA_END
#   LAYER_END
#   ..."LAYER:<layer_idx>" to "LAYER_END" are repeated for all tensors in the network...

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

path = "custom_weight.bin"
# print (net)
os.system ("echo ASPEN_DATA > " + path)
layer_idx = 1
# Dump the weights and biases of the model.
dump_tensor (path, net.conv1.weight, "LAYER:1\nTENSOR_TYPE:WEIGHT\n")
dump_tensor (path, net.conv1.bias, "LAYER:1\nTENSOR_TYPE:BIAS\n")
dump_tensor (path, net.conv2.weight, "LAYER:2\nTENSOR_TYPE:WEIGHT\n")
dump_tensor (path, net.conv2.bias, "LAYER:2\nTENSOR_TYPE:BIAS\n")
dump_tensor (path, net.fc1.weight, "LAYER:3\nTENSOR_TYPE:WEIGHT\n")
dump_tensor (path, net.fc1.bias, "LAYER:3\nTENSOR_TYPE:BIAS\n")
dump_tensor (path, net.fc2.weight, "LAYER:4\nTENSOR_TYPE:WEIGHT\n")
dump_tensor (path, net.fc2.bias, "LAYER:4\nTENSOR_TYPE:BIAS\n")
dump_tensor (path, net.fc3.weight, "LAYER:5\nTENSOR_TYPE:WEIGHT\n")
dump_tensor (path, net.fc3.bias, "LAYER:5\nTENSOR_TYPE:BIAS\n")

# Dump the inputs of the network.
np_arr = images.detach().numpy()
np_arr.astype('float32').tofile("custom_input.tensor")
