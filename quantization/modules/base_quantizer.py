import torch
from torch import nn, optim
import torch.nn.functional as F

class PerTensorQuantizer:
    def _quantize_signed_symmetric(self,weight:torch.Tensor,bits):
        q_min = -2**(bits - 1) -1
        q_max = 2 ** (bits - 1) - 1
        dtype = torch.int8 if bits == 8 else torch.int32

        max_val = weight.abs().max()
        if max_val == 0:
            scale = 1.0
            q_weight = torch.zeros_like(weight,dtype=dtype)
        else:
            scale = max_val / q_max
            q_weight = torch.round(weight / scale).clamp(q_min,q_max).to(dtype)
        return q_weight,scale
    
    def _quantize_unsigned_symmetric(self,tensor:torch.Tensor,bits):
        pass
    def _quantize_signed_asymmetric(self,tensor:torch.Tensor,bits):
        pass
    def _quantize_unsigned_asymmetric(self,tensor:torch.Tensor,bits):
        pass

    # Dequantization
    def _dequantize_signed_symmetric(self,q_tensor:torch.Tensor,scale:torch.float32):
        return q_tensor.to(torch.float32) * scale
    