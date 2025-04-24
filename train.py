# set the matplotlib backend so figures can be saved in the background
import matplotlib
from torchvision.transforms import ToTensor
import torch
from torch.optim import Adam
from torch.nn import NLLLoss
import os
import time


matplotlib.use("Agg")

from pyimagesearch import Lenet
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True, help='Path to output trained model')
parser.add_argument('-p', '--plot', type=str, required=True, help='Path to output loss/accuracy plot')
args = vars(parser.parse_args())

# define training hyperparameters
INIT_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 10
# define the train and val splits
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1 - TRAIN_SPLIT
# set the device we will be using to train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("[INFO] loading the MNIST dataset...")

os.makedirs('data', exist_ok=True)

train_data = MNIST(root='data', train=True, transform=ToTensor(), download=True)
test_data = MNIST(root='data', train=False, transform=ToTensor(), download=True)
print("Dataset downloaded successfully!")

num_train_split = int(len(train_data) * TRAIN_SPLIT)
num_val_split_not_int = len(train_data) * VAL_SPLIT
num_val_split = int(len(train_data) * VAL_SPLIT)
print(num_val_split_not_int, type(num_val_split_not_int))
print()
print(num_val_split, type(num_val_split))

generator = torch.Generator().manual_seed(42)

(train_data, val_data) = random_split(train_data, [num_train_split, num_val_split], generator=generator)

train_data_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=BATCH_SIZE)
test_data_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

train_steps = len(train_data_loader)
val_steps = len(val_data_loader)

print(f'train steps {train_steps}')
print(f'val steps {val_steps}')

model = Lenet(num_channels=1, classes=len(train_data.dataset.classes)).to(device)

print("[INFO] initializing the LeNet model...")
opt = Adam(params=model.parameters(), lr=INIT_LR)
loss = NLLLoss()

# initialize a dictionary to store training history
history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}
# measure how long training is going to take
print("[INFO] training the network...")
startTime = time.time()
