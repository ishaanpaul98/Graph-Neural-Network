import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import sklearn.metrics as metrics

BATCH_SIZE = 32

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(root="./data", train = True, download = True, transform = transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)

testset = torchvision.datasets.MNIST(root = "./data", train = False, download = True, transform = transform)

testloader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 2)

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3)
        self.d1 = nn.Linear(26 * 26 * 32, 128)
        self.d2 = nn.Linear(128, 10)

    def forward(self, x):
        # 32x1x28x28 -> 32x32x26x26
        x = self.conv1(x)
        x = F.relu(x)
        
        #Flatten -> 32 x (32x26x26)
        x = x.flatten(start_dim = 1)

        #32 x (32x26x26) -> 32*128
        x = self.d1(x)
        x = F.relu(x)

        #logits -> 32, 10
        logits = self.d2(x)
        out = F.softmax(logits, dim = 1)
        return out
    

