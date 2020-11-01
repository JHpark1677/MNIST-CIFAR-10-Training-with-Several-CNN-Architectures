import torch.nn as nn
import math
import torch
from torch.autograd import Variable
from tqdm.autonotebook import tqdm
import torch.nn.functional as F
import torch.nn.init as init

import torch.optim as optim
import torch.optim.lr_scheduler
import time
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

import torchvision
from torchvision import transforms as transforms
from torchvision.transforms import Compose, ToTensor, Normalize

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
from sklearn.datasets import load_digits
from sklearn import datasets, model_selection

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm


def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")
        

def _weights_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock, [3, 3, 3])


def resnet32():
    return ResNet(BasicBlock, [5, 5, 5])


def resnet44():
    return ResNet(BasicBlock, [7, 7, 7])


def resnet56():
    return ResNet(BasicBlock, [9, 9, 9])


def resnet110():
    return ResNet(BasicBlock, [18, 18, 18])


def resnet1202():
    return ResNet(BasicBlock, [200, 200, 200])


train_batch_size = 100
val_batch_size = 100
test_batch_size = 1000
train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()])
test_transform = transforms.Compose([transforms.ToTensor()])
train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=train_batch_size, shuffle=True)
val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
val_loader = torch.utils.data.DataLoader(dataset=val_set, batch_size=val_batch_size, shuffle=False)
test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_batch_size, shuffle=True)

start_ts = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet20().to(device)
epochs = 200
losses = []
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters())
train_loss_list = []
val_loss_list = []

batches = len(train_loader)
val_batches = len(val_loader)

for epoch in range(epochs):
    total_loss = 0.0
    progress = tqdm(enumerate(train_loader), total = batches)
    precision, recall, f1, accuracy = [], [], [], []
    model.train()

    for i, data in progress:
        train_x, train_y = data[0].to(device), data[1].to(device) # much time consuming way
        optimizer.zero_grad()
        outputs = model(train_x)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        if(i+1) % 100 == 0:
            model.eval()
            with torch.no_grad():
                val_losses = 0.0
                for j, data in enumerate(val_loader):
                    val_x, val_y = data[0].to(device), data[1].to(device) # much time consuming way
                    val_output = model(val_x)
                    v_loss = criterion(val_output, val_y)
                    val_losses += v_loss.item()
                    predicted_classes = torch.max(val_output, 1)[1]
                    for acc, metric in zip((precision, recall, f1, accuracy), 
                                          (precision_score, recall_score, f1_score, accuracy_score)):
                        acc.append(
                            calculate_metric(metric, val_y.cpu(), predicted_classes.cpu())
                        )
            train_loss_list.append(total_loss / 100)
            val_loss_list.append(val_losses / val_batches)
            temp_loss = total_loss
            total_loss = 0
        
    torch.cuda.empty_cache()
    for param_group in optimizer.param_groups:
        print("Current Learning rate is {}".format(param_group['lr']))
    print(f"Epoch {epoch+1}/{epochs}, training loss: {temp_loss/100}, validation loss: {val_losses/val_batches}")
    print_scores(precision, recall, f1, accuracy, batches)

print(f"Training time: {time.time()-start_ts}s")

#Evaluation
num = int(epochs * batches / 100)
iters = range(0, num)
plt.plot(iters, train_loss_list, 'g', label='Training loss')
plt.plot(iters, val_loss_list, 'b', label = 'Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('iters')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.savefig("CIFAR-10_ResNet_LrCurve.png")



precision2, recall2, f12, accuracy2 = [], [], [], []
precision3, recall3, f13, accuracy3 = [], [], [], []
result2 = []
result3 = []
test_Y = []
train_Y = []
test_loss = 0
with torch.no_grad():
    for k, data in enumerate(test_loader):
        test_x, test_y = data[0].to(device), data[1].to(device)
        test_output = model(test_x)
        t_loss = criterion(test_output, test_y)
        test_loss += t_loss.item()
        predicted_classes2 = torch.max(test_output, 1)[1]
        result2.append(predicted_classes2)
        test_Y.append(test_y)
        for acc2, metric2 in zip((precision2, recall2, f12, accuracy2), 
                                (precision_score, recall_score, f1_score, accuracy_score)):
            acc2.append(
                calculate_metric(metric2, test_y.cpu(), predicted_classes2.cpu())
            )
print("test_score")
print_scores(precision2, recall2, f12, accuracy2, len(test_loader))

train_loss = 0
with torch.no_grad():
    for h, data in enumerate(train_loader):
        train_x, train_y = data[0].to(device), data[1].to(device)
        train_output = model(train_x)
        train_loss = criterion(train_output, train_y)
        train_loss += t_loss.item()
        predicted_classes3 = torch.max(train_output, 1)[1]
        result3.append(predicted_classes3)
        train_Y.append(train_y)
        
        for acc3, metric3 in zip((precision3, recall3, f13, accuracy3), 
                                (precision_score, recall_score, f1_score, accuracy_score)):
            acc3.append(
                calculate_metric(metric3, train_y.cpu(), predicted_classes3.cpu())
            )

print("train_score")
print_scores(precision3, recall3, f13, accuracy3, len(train_loader))


## Precision Check
accumulated_result1 = []
accumulated_result1_y = []
accumulated_result2 = []
accumulated_result2_y = []
for i in range(0, len(result2)):
    accumulated_result1 = np.concatenate((accumulated_result1, result2[i].cpu().numpy()), axis = None)
    accumulated_result1_y = np.concatenate((accumulated_result1_y, test_Y[i].cpu().numpy()), axis = None)

for i in range(0, len(result3)):
    accumulated_result2 = np.concatenate((accumulated_result2, result3[i].cpu().numpy()), axis = None)
    accumulated_result2_y = np.concatenate((accumulated_result2_y, train_Y[i].cpu().numpy()), axis = None)

print("test precision score : ", precision_score(accumulated_result1_y, accumulated_result1, average = None))
print("train precision score : ", precision_score(accumulated_result2_y, accumulated_result2, average = None))

file = open("CIFAR-10_ResNet.txt","w")
file.write("Model's state_dict\n")

for param_tensor in model.state_dict():
    file.write("\n\n\n")
    file.write(param_tensor)
    file.write("\n")
    if len([model.state_dict()[param_tensor].cpu().numpy()]) != 1:
        for i in range(0, len(model.state_dict()[param_tensor])):
            file.write(np.array_str(model.state_dict()[param_tensor][i].cpu().numpy()))
    else : 
        file.write(np.array_str(model.state_dict()[param_tensor].cpu().numpy()))

file.close()
