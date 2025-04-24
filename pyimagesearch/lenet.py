from torch.nn import Module
from torch.nn import Conv2d, ReLU, MaxPool2d, Linear, Flatten, LogSoftmax

class Lenet(Module):
    def __init__(self, num_channels, classes):
        super(Lenet, self).__init__()
        # Let's define the layers

        #The first (CONV => RELU => POOL)
        self.conv_1 = Conv2d(in_channels=num_channels, out_channels=20, kernel_size=(5, 5))
        self.relu_1 = ReLU()
        self.max_pool_1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        #The second (CONV => RELU => POOL)
        self.conv_2 = Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))
        self.relu_2 = ReLU()
        self.max_pool_2 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        #The (FC => RELU)
        self.FC_1 = Linear(in_features=800, out_features=500)
        self.relu_fc = ReLU()

        self.FC_2 = Linear(in_features=500,out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.max_pool_1(x)

        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.max_pool_2(x)

        x = Flatten(x, 1)
        x = self.FC_1(x)

        x = self.FC_2(x)
        output = self.logSoftmax(x)

        return output
