import torch
from torch.autograd import Variable
from tqdm.autonotebook import tqdm
import torch.nn as nn
import torch.nn.functional as F
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

class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 96, kernel_size = 5, stride =1 , padding = 2, bias = True)
        self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2, bias = True)
        self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.fc1 = nn.Linear(2304,4096)
        self.fc2 = nn.Linear(4096, 10)
        self.fc3 = nn.Linear(10, 10)
        self.norm = nn.LocalResponseNorm(size= 5)

    def forward(self, x):
        x = F.max_pool2d(self.norm(F.relu(self.conv1(x))), kernel_size = 3, stride = 2)
        x = F.max_pool2d(self.norm(F.relu(self.conv2(x))), kernel_size = 3, stride = 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), kernel_size = 3, stride = 2)
        x = x.view(-1, 2304)
        x = F.dropout(F.relu(self.fc1(x)), 0.5)
        x = F.dropout(self.fc2(x), 0.5)
        return self.fc3(x)

def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")
        
        
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
model = AlexNet().to(device)
epochs = 200
losses = []
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters())
#scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[75,150], gamma=0.5)
train_loss_list = []
val_loss_list = []

batches = len(train_loader)
val_batches = len(val_loader)


for epoch in range(epochs):
    total_loss = 0.0
    #scheduler.step()
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
plt.savefig('AlexNet_Learning_Curve.png')



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


# file = open("CIFAR-10_AlexNet.txt","w")
# file.write("Model's state_dict\n")

# for param_tensor in model.state_dict():
#     file.write("\n\n\n")
#     file.write(param_tensor)
#     file.write("\n")
#     for i in range(0, len(model.state_dict()[param_tensor])):
#         file.write(np.array_str(model.state_dict()[param_tensor][i].cpu().numpy()))

# file.close()
