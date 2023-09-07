import torch
from torch import nn
import torchvision
from torchvision.transforms import transforms
import tqdm
from torchsummary import summary
from model import Lenet
import os
import torch.nn.functional as F


epoch = 10
checkpoint_dir = './checkpoint'

def test():
    net = Lenet()
    x = torch.randn(2, 1, 28, 28)
    y = net(x)
    print(y.size())


def train(epochs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=12)

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128, shuffle=False, num_workers=12)
    test_loader = iter(test_loader)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    net = Lenet(grayscale=True)

    net.to(device)
    summary(net, (1, 28, 28))

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    total_step = len(train_loader)
    acc = 0
    acc_last = 0
    for epoch in range(epochs):
        sum_loss = 0.0
        with tqdm.tqdm(total=total_step) as pbar:
            pbar.set_description('Epoch {}/{}'.format(epoch + 1, epochs))
            for i, data in enumerate(train_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logits = net(inputs)
                loss = loss_function(logits, labels)
                loss.backward()
                optimizer.step()

                sum_loss += loss.item()
                if i % 100 == 99:
                    test_imgs, test_labels = next(test_loader)
                    test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)
                    logits = net(test_imgs)
                    probas = F.softmax(logits, dim=1)
                    _, predict = torch.max(probas, 1)
                    correct_num = (predict == test_labels).sum()
                    # print('[%d, %d] loss: %.03f' % (epoch+1, i+1, sum_loss/100))
                    sum_loss = 0.0
                    pbar.set_postfix(loss=loss.item(), acc=correct_num.item() / test_labels.size(0))
                    acc_last = acc
                    acc = correct_num.item() / test_labels.size(0)

                pbar.update(1)
            if acc > acc_last:
                torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'lenet_best.pth'))

    torch.save(net.state_dict(), os.path.join(checkpoint_dir, 'lenet_last.pth'))
    net_ = net.eval().cpu()
    torch.onnx.export(net_, torch.randn([1, 1, 28, 28]), os.path.join(checkpoint_dir, 'lenet_last.onnx'), opset_version=10, do_constant_folding=True,
                      input_names=['input'], output_names=['output'])


if __name__ == '__main__':
    train(epoch)
