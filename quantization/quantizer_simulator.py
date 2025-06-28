import torch
import torch.nn as nn
from symmetric_quantizer import QuantConv2d,QuantLinear
def quantize_resnet_submodules(parent,bits:int=8,signed:bool=True):
    for name, module in parent.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(parent, name, QuantConv2d(module,bits,signed))
        elif isinstance(module, nn.Linear):
            setattr(parent, name, QuantLinear(module,bits,signed))
        else:
            quantize_resnet_submodules(module,bits,signed)

def quantize_resnet(model,bits:int=8,signed:bool=True):
    model_fp32 = model.eval()
    model_q = model_fp32
    for name,module in model_fp32.named_children():
        if isinstance(module,nn.Conv2d):
            setattr(model_q,name,QuantConv2d(module,bits,signed))
        elif isinstance(module,nn.Linear):
            setattr(model_q,name,QuantLinear(module,bits,signed))
        
        else:
            quantize_resnet_submodules(module,bits,signed)
    return model_q