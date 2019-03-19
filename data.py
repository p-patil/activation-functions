import torchvision
import torchvision.transforms as transforms

class Cifar10:
    def __init__(self, data_path="./data"):
        # Pre-processing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Fetch data
        self.trainset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=True,
            download=True,
            transform=self.transform)
        self.testset = torchvision.datasets.CIFAR10(
            root=data_path,
            train=False,
            download=True,
            transform=self.transform)

        # Define data loader
        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=4, shuffle=True, num_workers=2)
        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=4, shuffle=False, num_workers=2)

        # Cifar 10 classes
        self.classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog",
                        "horse", "ship", "truck")
