
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights
from torchvision import models
from modules import PerTensorQuantizer

# Quantizing Linear Layer
class QuantLinear(nn.Module,PerTensorQuantizer):
    def __init__(self,lin:nn.Linear,bits:int=8,signed:bool=True):
        super().__init__()
        self.weight = lin.weight
        self.bias = lin.bias
        self.qweight=None
        self.w_scale = None

        self.bits=bits
        self.signed=signed

    def clear(self):
        self.qweight=None
        self.w_scale=None        

    def forward(self,x:torch.Tensor):
        if self.qweight is None:
            self.qweight,self.w_scale = self._quantize_signed_symmetric(self.weight,bits=self.bits)
            self.qweight = self.qweight.t().to(torch.int32)

        qX,x_scale = self._quantize_signed_symmetric(x,bits=self.bits)
        qX = qX.to(torch.int32)
        
        #Linear Layer
        output = torch.mm(qX,self.qweight)
        
        #Dequantization
        y = self._dequantize_signed_symmetric(output,x_scale * self.w_scale)
        if self.bias is not None:
            y = y + self.bias
        
        return y
    
# Quantizing Convolutional Layer
class QuantConv2d(nn.Module,PerTensorQuantizer):
    def __init__(self,conv:nn.Conv2d,bits:int=8,signed:bool=True):
        super().__init__()

        # Normalize input
        kernel_size=conv.kernel_size
        stride=conv.stride
        padding=conv.padding
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weight=conv.weight
        self.bias=conv.bias

        # Quantized params
        self.qweight=None
        self.w_scale=None

        self.bits=bits
        self.signed=signed

    def forward(self,x:torch.Tensor):

        batch_size = x.size(0)
        
        if self.qweight is None:
            qW,self.w_scale = self._quantize_signed_symmetric(self.weight,bits=self.bits)
            self.qweight = qW.to(torch.int32)
   
        x_unfold = F.unfold(
            x,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        
        qx_unfold,x_scale = self._quantize_signed_symmetric(x_unfold,bits=self.bits)
        qx_unfold = qx_unfold.to(torch.int32)

        qw_flat = self.qweight.view(self.out_channels,-1)
        out = qw_flat @ qx_unfold
        out = self._dequantize_signed_symmetric(out,x_scale * self.w_scale)

        if self.bias is not None:
            out +=self.bias.view(1,-1,1) #[B,out_channel,L]

        # Reshape output to [B, out_channels, H_out, W_out]
        h_out = (x.shape[2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        w_out = (x.shape[3] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        out = out.view(batch_size, self.out_channels, h_out, w_out)
        
        return out

