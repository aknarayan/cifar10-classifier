import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

training_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, shuffle=False)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def imshow(img):
    img = img / 2 + 0.5  # un-normalise
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(training_loader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001)

for epoch in range(2):
    for i, data in enumerate(training_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print("Epoch {}, Image {}: loss = {}".format(epoch + 1, i, loss.item()))
print("Finished training")

ground_truths = []
predictions = []
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        for j in range(4):
            ground_truths.append(classes[labels[j]])
            predictions.append(classes[predicted[j]])
confusion_matrix = metrics.confusion_matrix(ground_truths, predictions)
accuracy = metrics.accuracy_score(ground_truths, predictions)
(p, r, f1, supp) = metrics.precision_recall_fscore_support(ground_truths, predictions, average='macro')
print("Confusion Matrix:")
print(confusion_matrix)
print("Accuracy: {}".format(accuracy))
print("Precision: {}".format(p))
print("Recall: {}".format(r))
print("F1 Score: {}".format(f1))
print("Support: {}".format(supp))