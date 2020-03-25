import torch
from net import Net
import torch.nn as nn
import torch.optim as optim
import sklearn.metrics as metrics


class Classifier:
    def __init__(self):
        self.net = Net()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=0.001)
        self.PATH = './cifar_net.pth'

    def train(self, training_loader, save_model):
        if save_model:
            for epoch in range(2):
                for i, data in enumerate(training_loader, 0):
                    inputs, labels = data
                    self.optimizer.zero_grad()
                    outputs = self.net(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()

                    if i % 1000 == 0:
                        print("Epoch {}, Image {}: loss = {}".format(epoch + 1, i, loss.item()))
            print("Finished training")
            torch.save(self.net.state_dict(), self.PATH)

    def evaluate(self, test_loader, classes, batch_size):
        self.net.load_state_dict(torch.load(self.PATH))
        ground_truths = []
        predictions = []
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                for j in range(batch_size):
                    ground_truths.append(classes[labels[j]])
                    predictions.append(classes[predicted[j]])
        confusion_matrix = metrics.confusion_matrix(ground_truths, predictions)
        accuracy = metrics.accuracy_score(ground_truths, predictions)
        (p, r, f1, _) = metrics.precision_recall_fscore_support(ground_truths, predictions, average='macro')
        print("Confusion Matrix:")
        print(confusion_matrix)
        print("Accuracy: {}".format(accuracy))
        print("Precision: {}".format(p))
        print("Recall: {}".format(r))
        print("F1 Score: {}".format(f1))
