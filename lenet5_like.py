import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet5_like(nn.Module):
    def __init__(self):
        super(LeNet5_like, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        # NOTE! This layer is different from LeNet5: we do not use the Gaussian connections for simplicity.
        self.fc3 = nn.Linear(84, 10)

    # Connect layers, define activation functions
    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Flat x for fc
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x): # To see dimensions of layers
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def predict(self, input):
        #input shape: round X player x 1 x 28 x 28
        nRound = input.shape[0]
        with torch.no_grad():
            # round X player x 28 x 28
            input = input.view(-1, 1, 28, 28)

            output = self(input)
            label = torch.argmax(output, 1)
            return label.view(nRound, 4).numpy()


