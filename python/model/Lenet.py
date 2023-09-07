import torch
from torch import nn


class Lenet(nn.Module):
    def __init__(self, num_classes=10, grayscale=False):
        super(Lenet, self).__init__()

        self.grayscale = grayscale
        self.num_classes = num_classes

        if self.grayscale:
            in_channels = 1
        else:
            in_channels = 3

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(20, 20, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(20 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            # nn.Sigmoid(),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        logits = self.classifier(x)
        # probas = F.softmax(logits, dim=1)

        return logits