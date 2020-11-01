from torchvision.models.resnet import ResNet, BasicBlock
from torchvision.datasets import MNIST
from tqdm.autonotebook import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import inspect
import time
import numpy as np
from torch import nn, optim
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torch.utils.data import DataLoader

import torch.optim.lr_scheduler
from torch.autograd import Variable
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm

class ResNet18(ResNet):
    def __init__(self):
        super(ResNet18, self).__init__(BasicBlock, [2, 2, 2, 2], num_classes=10)
        self.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        
    def forward(self, x):
        return (super(ResNet18, self).forward(x))


def get_data_loaders(train_batch_size, val_batch_size, test_batch_size):
    mnist = MNIST(download=True, train=True, root=".").train_data.float()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_transform = Compose([ Resize((224, 224)),ToTensor(), Normalize((mnist.mean()/255,), (mnist.std()/255,))])

    train_loader = DataLoader(MNIST(download=True, root=".", transform=data_transform, train=True),
                              batch_size=train_batch_size)

    val_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform, train=False),
                            batch_size=val_batch_size, shuffle=False)
    
    test_loader = DataLoader(MNIST(download=False, root=".", transform=data_transform,train=False), 
                             batch_size=test_batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


def calculate_metric(metric_fn, true_y, pred_y):
    if "average" in inspect.getfullargspec(metric_fn).args:
        return metric_fn(true_y, pred_y, average="macro")
    else:
        return metric_fn(true_y, pred_y)
    
def print_scores(p, r, f1, a, batch_size):
    for name, scores in zip(("precision", "recall", "F1", "accuracy"), (p, r, f1, a)):
        print(f"\t{name.rjust(14, ' ')}: {sum(scores)/batch_size:.4f}")


start_ts = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 5
model = ResNet18().to(device)
train_loader, val_loader, test_loader = get_data_loaders(100, 100, 1000)

losses = []
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(model.parameters())
#scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda = lambda epoch : 0.9 ** epoch)
train_loss_list = []
val_loss_list = []

batches = len(train_loader)
val_batches = len(val_loader)

# training loop + eval loop
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
plt.savefig("MNIST_ResNet_LrCurve.png")



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