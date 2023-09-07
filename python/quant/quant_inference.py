import torch
import torchvision
import tqdm
from quant.quant_model import Lenet, load_quant
from torchsummary import summary
from torchvision.transforms import transforms
import torch.nn.functional as F


def quant_inference():
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    net = Lenet(grayscale=True)

    scale_out = load_quant(net, './param/onnx_param.pth', './param/onnx_scale.txt')
    net.to(device)
    net.eval()
    summary(net, (1, 28, 28), device=device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=512, shuffle=False, num_workers=0)

    correct_num_sum = 0

    with tqdm.tqdm(total=len(test_loader)) as pbar:
        pbar.set_description('Test')
        for test_imgs, test_labels in test_loader:
            test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)
            logits = net(test_imgs)
            logits = (logits / scale_out).clamp_(-128, 127).floor_()
            logits = logits * scale_out
            probas = F.softmax(logits, dim=1)
            _, predict = torch.max(probas, 1)
            correct_num = (predict == test_labels).sum()
            correct_num_sum += correct_num.item()
            pbar.set_postfix(acc=correct_num.item() / test_labels.size(0))
            pbar.update(1)

    # print('acc: ', correct_num_sum / len(test_set))
    return correct_num_sum / len(test_set)


if __name__ == '__main__':
    quant_inference()
