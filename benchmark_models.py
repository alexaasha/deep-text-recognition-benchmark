import argparse
import datetime
import json
import os
import platform
import time

import pandas as pd
import psutil
import torch
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader

from model import Model
from utils import CTCLabelConverterForBaiduWarpctc, CTCLabelConverter, AttnLabelConverter

torch.backends.cudnn.benchmark = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
# This flag allows you to enable the inbuilt cudnn auto-tuner to find the best algorithm to use for your hardware.
# If you check it using the profile tool, the cnn method such as winograd, fft, etc. is used for the first iteration and the best operation is selected for the device.

precisions = ["float", "half", "double"]
# For post-voltaic architectures, there is a possibility to use tensor-core at half precision.
# Due to the gradient overflow problem, apex is recommended for practical use.

model_parser = argparse.ArgumentParser(description="PyTorch Benchmarking")
model_parser.add_argument('--eval_data', required=True, help='path to evaluation dataset')
model_parser.add_argument('--benchmark_all_eval', action='store_true', help='evaluate 10 benchmark evaluation datasets')
model_parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
model_parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
model_parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
""" Data processing """
model_parser.add_argument('--batch_max_length', type=int, default=25, help='maximum-label-length')
model_parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
model_parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')
model_parser.add_argument('--rgb', action='store_true', help='use rgb input')
model_parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
model_parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
model_parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
model_parser.add_argument('--data_filtering_off', action='store_true', help='for data_filtering_off mode')
model_parser.add_argument('--baiduCTC', action='store_true', help='for data_filtering_off mode')
""" Model Architecture """
model_parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
model_parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
model_parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
model_parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
model_parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
model_parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
model_parser.add_argument('--output_channel', type=int, default=512,
                          help='the number of output channel of Feature extractor')
model_parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

bench_opts = Namespace(WARM_UP=5, NUM_TEST=50, BATCH_SIZE=12,
                       NUM_CLASSES=1000, NUM_GPU=1, folder="./result")

model_opts = model_parser.parse_args()
bench_opts.BATCH_SIZE *= bench_opts.NUM_GPU


class RandomDataset(Dataset):
    def __init__(self, length):
        self.len = length
        self.data = torch.randn(model_opts.input_channel, model_opts.imgH, model_opts.imgW, length)

    def __getitem__(self, index):
        return self.data[:, :, :, index]

    def __len__(self):
        return self.len


rand_loader = DataLoader(
    dataset=RandomDataset(bench_opts.BATCH_SIZE * (bench_opts.WARM_UP + bench_opts.NUM_TEST)),
    batch_size=model_opts.batch_size,
    shuffle=False,
    num_workers=8,
)


def inference(precision="float"):
    benchmark = {}
    with torch.no_grad():
        model = torch.nn.DataParallel(Model(model_opts)).to(device)
        model.eval()
        durations = []
        model_name = f"{model_opts.Transformation}-{model_opts.FeatureExtraction}-" \
                     f"{model_opts.SequenceModeling}-" \
                     f"{model_opts.Prediction}"

        print(f"Precision: {precision}; Model: {model_name}")
        for step, img in enumerate(rand_loader):
            img = getattr(img, precision)()
            torch.cuda.synchronize()
            start = time.time()
            model(img.to(device), "random text")
            torch.cuda.synchronize()
            end = time.time()
            if step >= bench_opts.WARM_UP:
                durations.append((end - start) * 1000)
        print(
            f"{model_name} model average inference time : {sum(durations)/len(durations)}ms"
        )
        del model
        benchmark[model_name] = durations
    return benchmark


f"{platform.uname()}\n{psutil.cpu_freq()}\ncpu_count: {psutil.cpu_count()}\nmemory_available: {psutil.virtual_memory().available}"


if __name__ == "__main__":
    folder_name = bench_opts.folder

    device_name = f"{device}_{bench_opts.NUM_GPU}_gpus_"
    system_configs = f"{platform.uname()}\n\
                     {psutil.cpu_freq()}\n\
                    cpu_count: {psutil.cpu_count()}\n\
                    memory_available: {psutil.virtual_memory().available}"

    gpu_configs = [
        torch.cuda.device_count(),
        torch.version.cuda,
        torch.backends.cudnn.version(),
        torch.cuda.get_device_name(0),
    ]
    gpu_configs = list(map(str, gpu_configs))

    temp = [
        "Number of GPUs on current device : ",
        "CUDA Version : ",
        "Cudnn Version : ",
        "Device Name : ",
    ]

    os.makedirs(folder_name, exist_ok=True)
    with open(os.path.join(folder_name, "config.json"), "w") as f:
        json.dump(vars(bench_opts), f, indent=2)
    now = datetime.datetime.now()

    start_time = now.strftime("%Y/%m/%d %H:%M:%S")

    print(f"benchmark start : {start_time}")

    for idx, value in enumerate(zip(temp, gpu_configs)):
        gpu_configs[idx] = "".join(value)
        print(gpu_configs[idx])
    print(system_configs)

    with open(os.path.join(folder_name, "system_info.txt"), "w") as f:
        f.writelines(f"benchmark start : {start_time}\n")
        f.writelines("system_configs\n\n")
        f.writelines(system_configs)
        f.writelines("\ngpu_configs\n\n")
        f.writelines(s + "\n" for s in gpu_configs)

    if 'CTC' in model_opts.Prediction:
        if model_opts.baiduCTC:
            converter = CTCLabelConverterForBaiduWarpctc(model_opts.character)
        else:
            converter = CTCLabelConverter(model_opts.character)
    else:
        converter = AttnLabelConverter(model_opts.character)
    model_opts.num_class = len(converter.character)

    for precision in precisions:
        inference_result = inference(precision)
        inference_result_df = pd.DataFrame(inference_result)
        path = f"{folder_name}/{device_name}_{precision}_model_inference_benchmark.csv"
        inference_result_df.to_csv(path, index=False)

    now = datetime.datetime.now()

    end_time = now.strftime("%Y/%m/%d %H:%M:%S")
    print(f"benchmark end : {end_time}")
    with open(os.path.join(folder_name, "system_info.txt"), "a") as f:
        f.writelines(f"benchmark end : {end_time}\n")