import torch
import torch.nn as nn
from symmetric_quantizer import QuantConv2d,QuantLinear
def quantize_resnet_submodules(parent):
    for name, module in parent.named_children():
        if isinstance(module, nn.Conv2d):
            setattr(parent, name, QuantConv2d(module))
        elif isinstance(module, nn.Linear):
            setattr(parent, name, QuantLinear(module))
        else:
            quantize_resnet_submodules(module)

def quantize_resnet(model):
    model_fp32 = model.eval()
    model_q = model_fp32
    for name,module in model_fp32.named_children():
        if isinstance(module,nn.Conv2d):
            setattr(model_q,name,QuantConv2d(module))
        elif isinstance(module,nn.Linear):
            setattr(model_q,name,QuantLinear(module))
        
        else:
            quantize_resnet_submodules(module)
    return model_q