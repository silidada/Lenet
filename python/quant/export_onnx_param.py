import onnx
import torch
from onnx import numpy_helper
from model import *
import os
from collections import OrderedDict

print(os.getcwd())
quantized_model_path = '../checkpoint/quantized.onnx'
param_output_path = './param/onnx_param.pth'
scale_txt_path = './param/onnx_scale.txt'

if os.path.exists('./param') is False:
    os.mkdir('./param')


def export_onnx_params(onnx_path=quantized_model_path):
    model = Lenet(grayscale=True)
    onnx_model = onnx.load(onnx_path)

    # =================================get the key from onnx===========================
    w_keys = []
    b_keys = []
    for node in onnx_model.graph.node:
        for key in node.input:
            if 'weight' in key:
                w_keys.append(key)
                # print(key)
            elif 'bias' in key:
                b_keys.append(key)

    # ============get the param corresponding to keys above from onnx=====================
    w_data = []
    b_data = []
    for key in w_keys:
        for t in onnx_model.graph.initializer:
            if t.name == key:
                w_data.append(numpy_helper.to_array(t))
    for key in b_keys:
        for t in onnx_model.graph.initializer:
            if t.name == key:
                b_data.append(numpy_helper.to_array(t))

    # ========================construct keys and data into dict==============================
    params_dict = {}
    for idx, key in enumerate(w_keys):
        params_dict[key] = w_data[idx]
    for idx, key in enumerate(b_keys):
        params_dict[key] = b_data[idx]

    ckpt = model.state_dict()
    for key in ckpt.keys():
        if key in params_dict.keys():
            ckpt[key] = torch.Tensor(params_dict[key])

    torch.save(ckpt, param_output_path)
    print('onnx params had been saved to {}...'.format(param_output_path))
    assert model.load_state_dict(torch.load(param_output_path, map_location='cpu'))
    print('check onnx_param.pth successfully...')


def sort_func(x):
    if 'input' in x:
        return 1
    return int(x.split('_')[-1])


def get_onnx_scale(onnx_path=quantized_model_path):
    onnx_model = onnx.load(onnx_path)
    init_onnx = onnx_model.graph.initializer

    # dequant_dict = {}
    dequant_dict = OrderedDict()
    for node in onnx_model.graph.node:
        if node.op_type == 'QuantizeLinear':
            for t in init_onnx:
                # print(node.input[1])
                if t.name == node.input[1]:
                    # print(node.input[0])
                    dequant_dict[node.input[0]] = numpy_helper.to_array(t)
            # print(node.input[0])
    # print(dequant_dict.keys())
    with open(scale_txt_path, 'w') as f:
        f.write('{\n')
        activation_keys = []
        feature_keys = []
        classifier_keys = []
        for key in dequant_dict.keys():
            # print(key)
            if "PPQ_Variable" in key or "input" in key or "/" in key:
                activation_keys.append(key)
                # print(key)
                continue
            # f.write(f'\'{key}\': 1 / {dequant_dict[key]},\n')
            f.write(f'\"{key}\": {dequant_dict[key]},\n')
            if "features" in key and 'weight' in key:
                feature_keys.append(key)
            elif "classifier" in key and 'weight' in key:
                classifier_keys.append(key)
        # print(feature_keys)
        # print(classifier_keys)
        # activation_keys.sort(key=sort_func)
        # for key in activation_keys:
        #     # f.write(f'\'{key}\': 1 / {dequant_dict[key]},\n')
        #     # print(key)
        #     f.write(f'\"{key}\": {dequant_dict[key]},\n')

        for i in range(len(activation_keys)):
            if not i == len(activation_keys) - 1:
                f.write(f'\"{activation_keys[i]}\": {dequant_dict[activation_keys[i]]},\n')
            else:
                f.write(f'\"{activation_keys[i]}\": {dequant_dict[activation_keys[i]]}\n')

        f.write('}\n')

    print('onnx scales had been saved to ./working/onnx_scale.txt...')


if __name__ == '__main__':
    export_onnx_params()
    get_onnx_scale()
