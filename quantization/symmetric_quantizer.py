
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights
from torchvision import models


def quantize_symmetric(tensor: torch.Tensor,bits=8,signed=True):
    # Choose range
    if signed:
        qmin = -2 ** (bits - 1)
        qmax = 2 ** (bits - 1) - 1
        dtype = torch.int8 if bits == 8 else torch.int32
    else:
        qmin = 0
        qmax = 2 ** bits - 1
        dtype = torch.uint8 if bits == 8 else torch.int32
    
    # Avoid divide by zero
    max_val = tensor.abs().max()
    if max_val == 0:
        scale = 1.0
        q_tensor = torch.zeros_like(tensor, dtype=dtype)
    else:
        scale = max_val / qmax
        q_tensor = torch.round(tensor / scale).clamp(qmin, qmax).to(dtype)

    return q_tensor, scale

def dequantize_symmetric(q_tensor: torch.Tensor, scale: float):
    return q_tensor.to(torch.float32) * scale

class QuantConv2d(nn.Module):
    def __init__(self,conv:nn.Conv2d,bits:int=8,signed:bool=True):
        super().__init__()
        self.stride=conv.stride
        self.padding=conv.padding
        self.groups=conv.groups

        self.weight=conv.weight
        self.bias=conv.bias

        self.qweight=None
        self.w_scale=None

        self.bits=bits
        self.signed=signed

    def forward(self,x:torch.Tensor):
        if self.qweight is None:
            qW,self.w_scale = quantize_symmetric(self.weight,bits=self.bits,signed=self.signed)
            self.qweight = qW.to(torch.int32)
            #Dequantize
            self.w_deq = dequantize_symmetric(self.qweight,self.w_scale).to(x.device)

        qX,x_scale = quantize_symmetric(x,bits=self.bits,signed=self.signed)
        x_deq = dequantize_symmetric(qX,x_scale)
        

        # Float32 convolution
        y = F.conv2d(x_deq,self.w_deq,bias=self.bias,
                           stride=self.stride,
                           padding=self.padding,
                           groups=self.groups,
                           )
        
        return y


class QuantLinear(nn.Module):
    def __init__(self,lin:nn.Linear,bits:int=8,signed:bool=True):
        super().__init__()
        self.weight = lin.weight
        self.bias = lin.bias
        self.qweight=None
        self.w_scale = None

        self.bits=bits
        self.signed=signed

    def forward(self,x:torch.Tensor):
        if self.qweight is None:
            qW,self.w_scale = quantize_symmetric(self.weight,bits=self.bits,signed=self.signed)
            self.qweight = qW.to(torch.int32)
            #Dequantize
            self.w_deq = dequantize_symmetric(self.qweight,self.w_scale)
        
        qX,x_scale = quantize_symmetric(x,bits=self.bits,signed=self.signed)
        x_deq = dequantize_symmetric(qX,x_scale)
        

        

        y = F.linear(x_deq,self.w_deq,bias=self.bias)
        
        return y
    
