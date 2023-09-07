import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import transforms
import tqdm
from torchsummary import summary
import json


class Conv2d_quant(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, qmin=-128, qmax=127):
        super(Conv2d_quant, self).__init__(in_channels, out_channels, kernel_size, stride,
                                           padding, dilation, groups, bias)
        self.scale_w = None
        self.scale_b = None
        self.scale_x = None
        self.scale_out = None
        self.qmin = qmin
        self.qmax = qmax

    def load(self, weight, bias, scale_w=None, scale_b=None, scale_x=None):
        self.scale_w = scale_w
        self.scale_b = scale_b
        self.scale_x = scale_x

        if scale_w is not None:
            self.weight.data = (weight / scale_w).clamp_(self.qmin, self.qmax).floor_()
        else:
            self.weight.data = weight

        if scale_b is not None:
            self.bias.data = (bias / scale_b).clamp_(self.qmin, self.qmax).floor_()
        else:
            self.bias.data = bias

        if scale_w is not None and scale_x is not None:
            self.scale_out = self.scale_x * self.scale_w

            self.scale_out = torch.tensor(self.scale_out).to(self.weight.device)
            self.scale_x = torch.tensor(self.scale_x).to(self.weight.device)

    def forward(self, inputs):
        inputs = inputs / self.scale_x
        inputs = inputs.clamp_(self.qmin, self.qmax).floor_()

        outputs = super(Conv2d_quant, self).forward(inputs)

        outputs = outputs * self.scale_out

        return outputs


class Relu_quant(nn.ReLU):
    def __init__(self):
        super().__init__()


class MaxPool2d_quant(nn.MaxPool2d):
    def __init__(self, kernel_size):
        super().__init__(kernel_size=kernel_size)


class Flatten_quant(nn.Flatten):
    def __init__(self):
        super().__init__()


class Linear_quant(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, qmin=-128, qmax=127):
        super(Linear_quant, self).__init__(in_features, out_features, bias)
        self.scale_w = None
        self.scale_b = None
        self.scale_x = None
        self.scale_out = None
        self.qmin = qmin
        self.qmax = qmax

    def load(self, weight, bias, scale_w=None, scale_b=None, scale_x=None):
        self.scale_w = scale_w
        self.scale_b = scale_b
        self.scale_x = scale_x

        if scale_w is not None:
            self.weight.data = (weight / scale_w).clamp_(self.qmin, self.qmax).floor_()
        else:
            self.weight.data = weight

        if scale_b is not None:
            self.bias.data = (bias / scale_b).clamp_(self.qmin, self.qmax).floor_()
        else:
            self.bias.data = bias

        if scale_w is not None and scale_x is not None:
            self.scale_out = self.scale_x * self.scale_w
            self.scale_out = torch.tensor(self.scale_out).to(self.weight.device)
            self.scale_x = torch.tensor(self.scale_x).to(self.weight.device)

    def forward(self, inputs):
        inputs = inputs / self.scale_x
        inputs = inputs.clamp_(self.qmin, self.qmax).floor_()

        outputs = super(Linear_quant, self).forward(inputs)

        outputs = outputs * self.scale_out

        return outputs


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
            Conv2d_quant(in_channels, 20, kernel_size=5, stride=1),
            Relu_quant(),
            MaxPool2d_quant(kernel_size=2),
            Conv2d_quant(20, 20, kernel_size=5, stride=1),
            Relu_quant(),
            MaxPool2d_quant(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            Linear_quant(20 * 4 * 4, 120),
            Relu_quant(),
            Linear_quant(120, 84),
            Linear_quant(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.classifier(x)

        return x


def load_quant(net, param_path, scale_path):
    ckpt = torch.load(param_path, map_location='cpu')
    with open(scale_path, 'r', encoding="utf-8") as f:
        scale_dict = json.load(f)
    scale_x_list = []
    for key, value in scale_dict.items():
        if 'PPQ_Variable_' in key or 'input' in key:
            scale_x_list.append(value)
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weight = ckpt[name + '.weight']
            bias = ckpt[name + '.bias']
            scale_w = scale_dict[name + '.weight']
            scale_b = scale_dict[name + '.bias']
            scale_x = scale_x_list.pop(0)
            module.load(weight, bias, scale_w, scale_b, scale_x)

    return scale_x_list[0]


if __name__ == "__main__":
    net = Lenet(grayscale=True)
    load_quant(net, "./param/onnx_param.pth", "./param/onnx_scale.txt")
    print(net.features[0].scale_x)
    print(net.features[0].scale_w)
    print(net.features[0].scale_b)
    print(net.features[0].weight)
