from sklearn.metrics import accuracy_score
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout2d
from torch.optim import Adam, SGD
from fastai.vision.all import *
import matplotlib.pyplot as plt
import get_data

## Our froms cratch CNN
class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.cnn_layers = Sequential(
            # Defining a 2D convolution layer
            Conv2d(3, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            # Defining another 2D convolution layer ... 
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),

            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
            BatchNorm2d(4),
            ReLU(inplace=True),
            MaxPool2d(kernel_size=2, stride=2),
        )
        self.linear_layers = Sequential(
            Linear(2 * 4 * 2, 365)
        )

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
## Defining the model
model = Net()
## Defining the loss function
criterion = CrossEntropyLoss()
print(model)
## Line below used for smaller batch of images
# dls = ImageDataLoaders.from_folder("places365_test/train", valid_pct = 0.2,bs = 64, val_bs=12, device = device)
dls = ImageDataLoaders.from_folder("places365_standard", bs = 64, val_bs=64, device = device)
dls.valid.show_batch(max_n=4, nrows=1)
n_epochs = 25
## Empty list to store training losses
train_losses = []
## Empty list to store validation losses
val_losses = []
## Training with our from scratch model
# learn = Learner(dls, model, metrics=error_rate, loss_func=criterion, opt_func=Adam, lr = 0.1)
## Training with Resnet18
# learn.fit_one_cycle(n_epochs)
# learn1 = cnn_learner(dls, models.resnet18, metrics=error_rate, loss_func=criterion, opt_func=Adam, lr=.2)
# learn1.fit_one_cycle(n_epochs)
## Training with Resnet50
learn2 = cnn_learner(dls, models.resnet50, metrics=error_rate, loss_func=criterion, opt_func=Adam, lr=.2)
learn2.fit_one_cycle(n_epochs)