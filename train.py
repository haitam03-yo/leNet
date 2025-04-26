# set the matplotlib backend so figures can be saved in the background
import matplotlib
from torchvision.transforms import ToTensor
import torch
from torch.optim import Adam
from torch.nn import NLLLoss
import os
import time
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


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
EPOCHS = 3
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
lossFn = NLLLoss()

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

#I stoped here (CTR + F -> "Below follows our training loop:", https://pyimagesearch.com/2021/07/19/pytorch-training-your-first-convolutional-neural-network-cnn/)
for epoch in range(EPOCHS):
    print(f"epoch {epoch}..")
    model.train()

    # initialize the total training and validation loss
    totalTrainLoss = 0
    totalValLoss = 0
    # initialize the number of correct predictions in the training
    # and validation step
    trainCorrect = 0
    valCorrect = 0

    for (x, y) in train_data_loader:
        (x, y) = (x.to(device), y.to(device))
        opt.zero_grad()

        y_pred = model(x)
        loss = lossFn(y_pred, y)

        loss.backward()
        opt.step()

        totalTrainLoss += loss
        trainCorrect += (y_pred.argmax(1) == y).type(torch.float).sum().item()

    with torch.no_grad():
        model.eval()
        for (x, y) in val_data_loader:

            (x, y) = (x.to(device), y.to(device))
            y_pred_val = model(x)
            totalValLoss += lossFn(y_pred_val, y)

            valCorrect += (y_pred_val.argmax(1) == y).type(torch.float).sum().item()

    avg_training_loss = totalTrainLoss / train_steps
    avg_validation_loss = totalValLoss / val_steps

    trainCorrect = trainCorrect / len(train_data_loader.dataset)
    valCorrect = valCorrect / len(val_data_loader.dataset)

    # update our training history
    history["train_loss"].append(avg_training_loss.cpu().detach().numpy())
    history["train_acc"].append(trainCorrect)
    history["val_loss"].append(avg_validation_loss.cpu().detach().numpy())
    history["val_acc"].append(valCorrect)
    # print the model training and validation information
    print("[INFO] EPOCH: {}/{}".format(epoch + 1, EPOCHS))
    print("Train loss: {:.6f}, Train accuracy: {:.4f}".format(
        avg_training_loss, trainCorrect))
    print("Val loss: {:.6f}, Val accuracy: {:.4f}\n".format(
        avg_validation_loss, valCorrect))

# finish measuring how long training took
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
    endTime - startTime))
# we can now evaluate the network on the test set
print("[INFO] evaluating network...")
# turn off autograd for testing evaluation
with torch.no_grad():
    # set the model in evaluation mode
    model.eval()

    # initialize a list to store our predictions
    preds = []
    # loop over the test set
    for (x, y) in test_data_loader:
        # send the input to the device
        x = x.to(device)
        # make the predictions and add them to the list
        pred = model(x)
        preds.extend(pred.argmax(axis=1).cpu().numpy())

# generate a classification report
print(classification_report(test_data.targets.cpu().numpy(),
                            np.array(preds), target_names=test_data.classes))

output_dir_model = os.path.dirname(args['model'])
output_dir_plot = os.path.dirname(args['plot'])

os.makedirs(output_dir_plot, exist_ok=True)
os.makedirs(output_dir_model, exist_ok=True)

plt.style.use('ggplot')
plt.figure()
plt.plot(history['train_acc'], label="train accuracy")
plt.plot(history['val_acc'], label="val_acc")
plt.plot(history['train_loss'], label="train loss")
plt.plot(history['val_loss'], label="val loss")
plt.title("graph accuracy, loss train/val set")
plt.xlabel('% Epochs')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.savefig(args['plot'])


torch.save(model, args['model'])
