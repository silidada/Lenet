from typing import Iterable, Tuple
from ppq import (BaseGraph, QuantizationSettingFactory, TargetPlatform,
                 convert_any_to_numpy, torch_snr_error)
from ppq.api import (dispatch_graph, export_ppq_graph, load_onnx_graph,
                     quantize_onnx_model)
from ppq.executor.torch import TorchExecutor
from ppq.quantization.analyse.graphwise import graphwise_error_analyse
from quant.export_onnx_param import *
from quant.quant_inference import *
import onnxruntime
import onnx
import tqdm


DEVICE = 'cuda'
QUANT_PLATFORM = TargetPlatform.FPGA_INT8
ONNX_PATH = "./checkpoint/lenet_last.onnx"
ONNX_OUTPUT_PATH = "./checkpoint/quantized.onnx"


def read_image(data_loader, device) -> torch.Tensor:
    data = next(iter(data_loader))
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)
    return inputs


# ------------------------------------------------------------
# 在这个例子中我们将向你展示如何量化一个 onnx 模型，执行误差分析，并与 onnxruntime 对齐结果
# 在这个例子中，我们特别地为你展示如何量化一个多输入的模型
# 此时你的 Calibration Dataset 应该是一个 list of dictionary
# ------------------------------------------------------------
def generate_calibration_dataset(data_loader, num_of_batches: int = 256) -> Tuple[Iterable[dict], torch.Tensor]:
    dataset = []
    for i in range(num_of_batches):
        sample = {'input': read_image(data_loader, "cuda")}
        dataset.append(sample)
    return dataset, sample  # last sample


def collate_fn(batch: dict) -> torch.Tensor:
    # print(batch)
    return batch


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])
test_set = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

# ------------------------------------------------------------
# 在这里，我们仍然创建一个 QuantizationSetting 对象用来管理量化过程
# 我们将调度方法修改为 conservative，并且要求 PPQ 启动量化微调
# ------------------------------------------------------------
QSetting = QuantizationSettingFactory.default_setting()
QSetting.lsq_optimization = False
QSetting.ssd_equalization = False
QSetting.equalization = False

# ------------------------------------------------------------
# 准备好 QuantizationSetting 后，我们加载模型，并且要求 ppq 按照规则完成图调度
# ------------------------------------------------------------
graph = load_onnx_graph(onnx_import_file=ONNX_PATH)
graph = dispatch_graph(graph=graph, platform=QUANT_PLATFORM)

print(graph.inputs)

if len(graph.outputs) != 1:
    raise ValueError('This Script Requires graph to have only 1 output.')

# ------------------------------------------------------------
# 生成校准所需的数据集，我们准备开始完成网络量化任务
# ------------------------------------------------------------
calibration_dataset, sample = generate_calibration_dataset(data_loader=test_loader, num_of_batches=256)
quantized = quantize_onnx_model(
    onnx_import_file=ONNX_PATH, calib_dataloader=calibration_dataset,
    calib_steps=32, input_shape=None, inputs=collate_fn(sample),
    setting=QSetting, collate_fn=collate_fn, platform=QUANT_PLATFORM,
    device=DEVICE, verbose=1)

# ------------------------------------------------------------
# 在 PPQ 完成网络量化之后，我们特别地保存一下 PPQ 网络执行的结果
# 在本样例的最后，我们将对比 PPQ 与 Onnxruntime 的执行结果是否相同
# ------------------------------------------------------------
executor, reference_outputs = TorchExecutor(quantized), []
for sample in calibration_dataset:
    reference_outputs.append(executor.forward(collate_fn(sample)))

# ------------------------------------------------------------
# 执行网络误差分析，并导出计算图
# ------------------------------------------------------------
graphwise_error_analyse(
    graph=quantized, running_device=DEVICE,
    collate_fn=collate_fn, dataloader=calibration_dataset)

export_ppq_graph(graph=quantized, platform=TargetPlatform.ONNXRUNTIME,
                 graph_save_to=ONNX_OUTPUT_PATH)

# ------------------------------------------------------------
# 导出量化因子以及模型参数
# ------------------------------------------------------------
export_onnx_params(onnx_path=ONNX_OUTPUT_PATH)
get_onnx_scale(onnx_path=ONNX_OUTPUT_PATH)

# ------------------------------------------------------------
# 使用量化后的模型进行推理
# ------------------------------------------------------------
quant_acc = quant_inference()


# ------------------------------------------------------------
# 使用 onnxruntime 进行推理原模型
# ------------------------------------------------------------
model_orin = onnx.load(ONNX_PATH)
sess = onnxruntime.InferenceSession(model_orin.SerializeToString(), providers=['CUDAExecutionProvider'])
test_set = torchvision.datasets.MNIST(root='../data', train=False, download=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
test_loader = iter(test_loader)
correct_num_sum = 0
with tqdm.tqdm(total=len(test_loader)) as pbar:
    pbar.set_description('Test orin')
    for i in range(len(test_loader)):
        test_data, test_label = next(test_loader)
        result = sess.run(
            None,
            {'input': test_data.numpy()}
        )
        probas = F.softmax(torch.tensor(result[0]), dim=1)
        _, predict = torch.max(probas, 1)
        correct_num = (predict == test_label).sum()
        correct_num_sum += correct_num.item()
        pbar.set_postfix(acc=correct_num.item() / test_label.size(0))
        pbar.update(1)
orin_acc = correct_num_sum/len(test_set)

print("-"*50)
print('quant_acc: \t\t', quant_acc)
print('orin_acc: \t\t', orin_acc)
print("acc decrease:\t %.4f" % (orin_acc - quant_acc))
print("-"*50)
