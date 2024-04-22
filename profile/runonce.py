import torch
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from optimized.rmsnorm import RMSNormTriton

seq_len = 4096
model_dim = 4096
x = torch.randn(seq_len, model_dim, device='cuda', dtype=torch.float32)
rmsnorm_test = RMSNormTriton(dim=model_dim).cuda()
y = rmsnorm_test.forward(x)

print("Completed")