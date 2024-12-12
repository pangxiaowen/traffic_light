import argparse
import sys
import time

sys.path.append('../')  # to run '$ python *.py' files in subdirectories
import onnx
import models
from models.experimental import attempt_load
from utils.activations import Hardswish, SiLU
from utils.general import set_logging, check_img_size
from utils.torch_utils import select_device
import torch

device = select_device("cpu")
model = attempt_load(weights="/home/pxw/Downloads/traffic_data/yolov5/runs/train/exp9/weights/best.pt", device=device,inplace=True, fuse=True)

output_names=["p3", "p4", "p5"]
im = torch.zeros(1, 3, 640, 640).to(device)
model.to(device)
model.eval()

torch.onnx.export(
    model,  # --dynamic only compatible with cpu
    im,
    "asp_yolov5.onnx",
    verbose=False,
    opset_version=13,
    do_constant_folding=True,  # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
    input_names=["images"],
    output_names=output_names,
)

onnx_path = "./asp_yolov5.onnx"
model_onnx = onnx.load(onnx_path)  # load onnx model
onnx.checker.check_model(model_onnx)  # check onnx model

# add yolov5_decoding:
import onnx_graphsurgeon as onnx_gs
import numpy as np
yolo_graph = onnx_gs.import_onnx(model_onnx)

p3 = yolo_graph.outputs[0]
p4 = yolo_graph.outputs[1]
p5 = yolo_graph.outputs[2]

decode_out_0 = onnx_gs.Variable(
    "DecodeNumDetection",
    dtype=np.int32
)
decode_out_1 = onnx_gs.Variable(
    "DecodeDetectionBoxes",
    dtype=np.float32
)
decode_out_2 = onnx_gs.Variable(
    "DecodeDetectionScores",
    dtype=np.float32
)
decode_out_3 = onnx_gs.Variable(
    "DecodeDetectionClasses",
    dtype=np.int32
)

decode_attrs = dict()

decode_attrs["max_stride"] = int(max(model.stride))
decode_attrs["num_classes"] = model.model[-1].nc
decode_attrs["anchors"] = [float(v) for v in [10,13, 16,30, 33,23, 30,61, 62,45, 59,119, 116,90, 156,198, 373,326]]
decode_attrs["prenms_score_threshold"] = 0.25

decode_plugin = onnx_gs.Node(
    op="YoloLayer_TRT",
    name="YoloLayer",
    inputs=[p3, p4, p5],
    outputs=[decode_out_0, decode_out_1, decode_out_2, decode_out_3],
    attrs=decode_attrs
)

yolo_graph.nodes.append(decode_plugin)
yolo_graph.outputs = decode_plugin.outputs
yolo_graph.cleanup().toposort()
model_onnx = onnx_gs.export_onnx(yolo_graph)

d = {'stride': int(max(model.stride)), 'names': model.names}
for k, v in d.items():
    meta = model_onnx.metadata_props.add()
    meta.key, meta.value = k, str(v)

onnx.save(model_onnx, "quant_plugin.onnx")