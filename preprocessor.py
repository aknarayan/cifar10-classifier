import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms


class Preprocessor:
    def __init__(self, batch_size):
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.training_set = None
        self.training_loader = None
        self.test_set = None
        self.test_loader = None
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.batch_size = batch_size

    def load(self):
        self.training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                         transform=self.transform)
        self.training_loader = torch.utils.data.DataLoader(self.training_set, batch_size=self.batch_size, shuffle=True)

        self.test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                     transform=self.transform)
        self.test_loader = torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

    def imshow(self, img):
        img = img / 2 + 0.5  # un-normalise
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def show_examples(self):
        data_iter = iter(self.training_loader)
        images, labels = data_iter.next()
        self.imshow(torchvision.utils.make_grid(images))
        print(' '.join('%5s' % self.classes[labels[j]] for j in range(self.batch_size)))
