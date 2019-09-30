import torch
import torch.nn.functional as F

from model import SimpleConvNet
from data import Cifar10

def test(net, data):
    correct = total = 0
    with torch.no_grad():
        for images, labels in data.testloader:
            outputs = net(images)
            _, predicted = torch.max(
                outputs.data,
                dim=1)  # TODO(piyush) See why the ".data" is necessary
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct, total


def main():
    activations = {
        "relu": F.relu,
        "leaky_relu": F.leaky_relu,
        "log_sigmoid": F.logsigmoid,
        "softsign": F.softsign,
        "tanh": F.tanh,
        "sigmoid": F.sigmoid
    }

    loss_histories = {}
    for activation_name, activation in activations.items():
        print("Activation function {}".format(activation_name))

        net = SimpleConvNet(activation=activation)
        data = Cifar10()

        # Use GPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

        # Train
        print("Training for activation function {}".format(activation_name))
        loss_history = net.train(data.trainloader, device=device)
        loss_histories[activation_name] = loss_history

        # Test
        correct, total = test(net, data)
        print("Test accuracy: {}%".format(100 * correct / total))

    # TODO(piyush) Plot directly
    print("Done. Dumping loss_histories.pkl")
    import pickle
    with open("loss_histories.pkl", "wb") as f:
        pickle.dump(loss_histories, f)


if __name__ == "__main__":
    main()
