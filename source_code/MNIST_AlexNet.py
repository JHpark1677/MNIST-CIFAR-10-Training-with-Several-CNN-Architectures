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
        self.conv1 = nn.Conv2d(1, 96, kernel_size = 5, stride =1 , padding = 2, bias = True)
        self.conv2 = nn.Conv2d(96, 256, kernel_size = 5, stride = 1, padding = 2, bias = True)
        self.conv3 = nn.Conv2d(256, 384, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv4 = nn.Conv2d(384, 384, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.conv5 = nn.Conv2d(384, 256, kernel_size = 3, stride = 1, padding = 1, bias = True)
        self.fc1 = nn.Linear(1024,2304)
        self.fc2 = nn.Linear(2304, 10)
        self.fc3 = nn.Linear(10, 10)
        self.norm = nn.LocalResponseNorm(size= 5)

    def forward(self, x):
        x = F.max_pool2d(self.norm(F.relu(self.conv1(x))), kernel_size = 3, stride = 2)
        x = F.max_pool2d(self.norm(F.relu(self.conv2(x))), kernel_size = 3, stride = 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(F.relu(self.conv5(x)), kernel_size = 3, stride = 2)
        x = x.view(-1, 1024)
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


start_ts = time.time()

mnist = datasets.fetch_openml('mnist_784', data_home="mnist_784")
mnist_label = mnist.target
mnist_data = mnist.data / 255

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_size = 50000
val_size = 10000
test_size = 10000


train_X, test_X, train_Y, test_Y = model_selection.train_test_split(mnist_data, mnist_label, train_size = 60000, test_size = test_size)
train_X, val_X, train_Y, val_Y = model_selection.train_test_split(train_X, train_Y, train_size = train_size, test_size = val_size)

val_X = val_X.reshape((len(val_X),1,28,28))
train_X = train_X.reshape((len(train_X),1,28,28))
test_X = test_X.reshape((len(test_X),1,28,28))

val_X = torch.tensor(val_X, dtype=torch.float)
val_Y = torch.tensor([int(x) for x in val_Y])
train_X = torch.tensor(train_X, dtype=torch.float)
train_Y = torch.tensor([int(x) for x in train_Y])
test_X = torch.tensor(test_X, dtype=torch.float)
test_Y = torch.tensor([int(x) for x in test_Y])

if torch.cuda.is_available() : 
    train_X = train_X.to(device)
    train_Y = train_Y.to(device)
    test_X = test_X.to(device)
    test_Y = test_Y.to(device)
    val_X = val_X.to(device)
    val_Y = val_Y.to(device)

# data is already loaded to gpu, so they warped with DataLoader in gpu
train = TensorDataset(train_X, train_Y)
val = TensorDataset(val_X, val_Y)
train_loader = DataLoader(train, batch_size = 100, shuffle = True)
val_loader = DataLoader(val, batch_size = 100, shuffle = True)

model = AlexNet()

# Parameter configuration
model = model.to(device)
criterion = nn.CrossEntropyLoss()
learning_rate = 0.001
#optimizer = optim.Adadelta(model.parameters())
optimizer = optim.Adam(model.parameters(), lr = learning_rate)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda = lambda epoch : 0.9 ** epoch)
train_loss_list = []
val_loss_list = []
batches = len(train_loader)
val_batches = len(val_loader)
epochs = 12

# Learning mode starts
for epoch in range(epochs):
    total_loss = 0.0
    scheduler.step()
    progress = tqdm(enumerate(train_loader), total = batches)
    precision, recall, f1, accuracy = [], [], [], []
    model.train()

    for i, data in progress:
        train_x, train_y = data
        optimizer.zero_grad()
        output = model(train_x)
        loss = criterion(output, train_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        #progress.set_description("Loss : {:.4f}".format(total_loss/(i+1)))

        if(i+1) % 100 == 0 :
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for j, val in enumerate(val_loader):
                    val_x, val_y = val
                    val_output = model(val_x)
                    v_loss = criterion(val_output, val_y)
                    val_loss += v_loss.item()
                    predicted_classes = torch.max(val_output, 1)[1]
                    for acc, metric in zip((precision, recall, f1, accuracy), 
                                           (precision_score, recall_score, f1_score, accuracy_score)):
                        acc.append(
                            calculate_metric(metric, val_y.cpu(), predicted_classes.cpu())
                        )
            train_loss_list.append(total_loss / 100)
            val_loss_list.append(val_loss / len(val_loader))
            temp_loss = total_loss
            total_loss = 0

    torch.cuda.empty_cache()
    for param_group in optimizer.param_groups:
        print("Current Learning rate is : {}".format(param_group['lr']))
    print(f"Epoch {epoch+1}/{epochs}, training loss : {temp_loss / 100}, validation loss : {val_loss / val_batches}")
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

plt.savefig("MNIST_AlexNet_LrCurve.png")


test = TensorDataset(test_X, test_Y)
test_loader = DataLoader(test, batch_size = 1000, shuffle = False)
train_loader = DataLoader(train, batch_size = 1000, shuffle = False)

precision2, recall2, f12, accuracy2 = [], [], [], []
precision3, recall3, f13, accuracy3 = [], [], [], []
result2 = []
result3 = []

test_loss = 0
with torch.no_grad():
    for k, data in enumerate(test_loader):
        test_x, test_y = data
        test_output = model(test_x)
        t_loss = criterion(test_output, test_y)
        test_loss += t_loss.item()
        predicted_classes2 = torch.max(test_output, 1)[1]
        result2.append(predicted_classes2)
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
        train_x, train_y = data
        train_output = model(train_x)
        train_loss = criterion(train_output, train_y)
        train_loss += t_loss.item()
        predicted_classes3 = torch.max(train_output, 1)[1]
        result3.append(predicted_classes3)
        for acc3, metric3 in zip((precision3, recall3, f13, accuracy3), 
                                (precision_score, recall_score, f1_score, accuracy_score)):
            acc3.append(
                calculate_metric(metric3, train_y.cpu(), predicted_classes3.cpu())
            )

print("train_score")
print_scores(precision3, recall3, f13, accuracy3, len(train_loader))


## Precision Check
accumulated_result1 = []
accumulated_result2 = []
for i in range(0, len(result2)):
    accumulated_result1 = np.concatenate((accumulated_result1, result2[i].cpu().numpy()), axis = None)

for i in range(0, len(result3)):
    accumulated_result2 = np.concatenate((accumulated_result2, result3[i].cpu().numpy()), axis = None)


print("test precision score : ", precision_score(test_Y.cpu().numpy(), accumulated_result1, average = None))
print("train precision score : ", precision_score(train_Y.cpu().numpy(), accumulated_result2, average = None))