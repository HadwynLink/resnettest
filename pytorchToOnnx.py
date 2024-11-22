import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision.models as models
import os
from skimage import io
#import pandas as pd
import myResnet
import onnx
from torch.export import export, ExportedProgram
from executorch.exir import to_edge

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(device)

model = myResnet.ResNet(myResnet.ResBlock, [3,4,6,3])
model.load_state_dict(torch.load('modelSave.pth', weights_only=True))
model.eval()

dummy_input = (torch.randn(1, 3, 256, 256),)

#aten_dialect: ExportedProgram = export(model, dummy_input)

# 2. to_edge: Make optimizations for Edge devices
#edge_program = to_edge(aten_dialect)

#executorch_program = edge_program.to_executorch()

# 4. Save the compiled .pte program
#with open("executorchtest.pte", "wb") as file:
#    file.write(executorch_program.buffer)


#onnx_program = torch.onnx.dynamo_export(model, dummy_input)


onnx_program = torch.onnx.export(model, 
                                 dummy_input, 
                                 "lasttest.onnx", 
                                 opset_version=18, 
                                 verbose=False, 
                                 do_constant_folding=True, 
                                 export_params=True)
torch.onnx