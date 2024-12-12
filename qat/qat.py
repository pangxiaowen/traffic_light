import argparse
import sys
import time

sys.path.append('../')  # to run '$ python *.py' files in subdirectories

import torch
import torch.nn as nn

import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
from torch.nn.parameter import Parameter
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
from pytorch_quantization.tensor_quant import QuantDescriptor
from absl import logging as quant_logging
from mmcv.cnn.utils.fuse_conv_bn import _fuse_conv_bn
from tqdm import tqdm
from typing import Callable
from pytorch_quantization.nn.modules.quant_conv import QuantConv2d, QuantConvTranspose2d    
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer
from utils.dataloaders import create_dataloader
from models.yolo import ClassificationModel, Detect, DetectionModel, SegmentationModel

def initialize():
    quant_logging.set_verbosity(quant_logging.ERROR)
    quant_desc_input = QuantDescriptor(calib_method="histogram")
    for item in quant_modules._DEFAULT_QUANT_MAP:
        item.replace_mod.set_default_quant_desc_input(quant_desc_input)

    quant_logging.set_verbosity(quant_logging.ERROR)
    
def transfer_torch_to_quantization(nninstance : torch.nn.Module, quantmodule):

    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            #quant_desc_input = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True)
            self.init_quantizer(quant_desc_input)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            # Turn on torch_hist to enable higher calibration speeds
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance

def replace_to_quantization_module(model : torch.nn.Module):

    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    def recursive_and_replace_module(module, prefix=""):
        for name in module._modules:
            submodule = module._modules[name]
            path      = name if prefix == "" else prefix + "." + name

            recursive_and_replace_module(submodule, path)

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:  
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

    recursive_and_replace_module(model)

def fuse_conv_bn(module):
    last_conv = None
    last_conv_name = None

    for name, child in module.named_children():
        if isinstance(child,
                      (nn.modules.batchnorm._BatchNorm, nn.SyncBatchNorm)):
            if last_conv is None:  # only fuse BN that is after Conv
                continue
            fused_conv = _fuse_conv_bn(last_conv, child)
            module._modules[last_conv_name] = fused_conv
            # To reduce changes, set BN as Identity instead of deleting it.
            module._modules[name] = nn.Identity()
            last_conv = None
        elif isinstance(child, QuantConv2d) or isinstance(child, nn.Conv2d): # or isinstance(child, QuantConvTranspose2d):
            last_conv = child
            last_conv_name = name
        else:
            fuse_conv_bn(child)
    return module


def calibrate_model(model : torch.nn.Module, dataloader, device, batch_processor_callback: Callable = None, num_batch=1):

    def compute_amax(model, **kwargs):
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        module.load_calib_amax(strict=False)
                    else:
                        module.load_calib_amax(strict=False, **kwargs)

                    module._amax = module._amax.to(device)
        
    def collect_stats(model, data_loader, device, num_batch=200):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()
                    
        pbar = enumerate(train_loader)
        nb = len(train_loader)           
        pbar = tqdm(pbar, total=nb)
        
        with torch.no_grad():
            for i, (imgs, targets, paths, _) in pbar:
                imgs = imgs.to(device, non_blocking=True).float() / 255 
                pred = model(imgs) 
    
        # for i in range(num_batch):
        #     with torch.no_grad():
        #         for X, y in data_loader:
        #             print(X)
        #             X.to(device)
        #             model(X)
        #     i+=1 
                    
        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    collect_stats(model, dataloader, device, num_batch=num_batch)
    compute_amax(model, method="mse")

def print_quantizer_status(module):
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print('TensorQuantizer name:{} disabled staus:{} module:{}'.format(name, module._disabled, module))
    
def set_quantizer_fast(module): 
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
             if isinstance(module._calibrator, calib.HistogramCalibrator):
                module._calibrator._torch_hist = True

if __name__ == "__main__":
    device = select_device("0")
    model = attempt_load(weights="/home/pxw/Downloads/traffic_data/yolov5/runs/train/exp10/weights/best.pt", device=device,inplace=True, fuse=True)
   
    initialize()
    replace_to_quantization_module(model)

    model.to(device)
    model.eval()
    
    # 加载训练数据集
    train_loader, dataset = create_dataloader(
        "/home/pxw/Downloads/traffic_data/actual_data/images/train",
        640,
        16,
        32
    )
    
    set_quantizer_fast(model)
    # calibrate_model(model, train_loader, device, None, 10)
    print_quantizer_status(model)
    
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = True
            m.dynamic = False
            m.export = True
    
    output_names=["p3", "p4", "p5"]
    im = torch.zeros(1, 3, 640, 640).to(device)

    TensorQuantizer.use_fb_fake_quant = True
    torch.onnx.export(
        model,  # --dynamic only compatible with cpu
        im,
        "quant_yolo.onnx",
        verbose=False,
        opset_version=13,
        do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
        input_names=["images"],
        output_names=output_names,
    )
    
    torch.save(model.state_dict(), "model.pth")
    print("Save Pytorch Model State to model.pth")