import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# The i-th tuple represents the dimensions of the i-th convolutional layer,
# formatted as (in_channels, out_channels, kernel_size).
DEFAULT_CONV_ARCH = [(3, 6, 5), (6, 16, 5)]

# The i-th tuple represents the dimensions of the i-th fully connected layer,
# formatted as (in_features, out_features).
DEFAULT_FC_ARCH = [(16 * 5 * 5, 120), (120, 84), (84, 10)]

class SimpleConvNet(nn.Module):
    def __init__(self,
                 conv_arch=DEFAULT_CONV_ARCH,
                 fc_arch=DEFAULT_FC_ARCH,
                 pool=nn.MaxPool2d(2, 2),
                 activation=F.relu):
        super(SimpleConvNet, self).__init__()
        self.activation = activation
        self.conv_arch = conv_arch
        self.fc_arch = fc_arch
        self.pool = pool

        self.conv = [nn.Conv2d(*shape) for shape in conv_arch]
        self.fc = [nn.Linear(*shape) for shape in fc_arch]

        # Register modules
        for i, layer in enumerate(self.conv, 1):
            self.add_module("conv{}".format(i), layer)
        for i, layer in enumerate(self.fc, 1):
            self.add_module("fc{}".format(i), layer)

    def forward(self, x):
        for conv_layer in self.conv:
            x = self.pool(self.activation(conv_layer(x)))

        conv_out_dim = self.conv_arch[-1][1] * self.conv_arch[-1][2]**2
        x = x.view(-1, conv_out_dim)

        for fc_layer in self.fc[:-1]:
            x = self.activation(fc_layer(x))

        x = self.fc[-1](x)
        return x

    def train(self,
              trainloader,
              criterion=nn.CrossEntropyLoss(),
              optimizer=None,
              num_epochs=5,
              log_freq=2000,
              lr=0.001,
              momentum=0.9,
              device="cpu"):
        if optimizer is None:
            optimizer = optim.SGD(self.parameters(), lr=lr, momentum=momentum)
        else:
            optimizer = optimizer(self.parameters())

        loss_history = []
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                loss_history.append((epoch, i, loss.item()))
                running_loss += loss.item()
                if i % log_freq == 0:
                    print("Epoch {}, iteration {}: loss = {}".format(
                        epoch + 1, i, running_loss / log_freq))
                    running_loss = 0.0

        print("Done")
        return loss_history
