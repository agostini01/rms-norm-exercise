import torch

import torch.nn as nn

class MyRMSNorm(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-6):
		super().__init__()
		self.eps = eps
		# The gamma parameter
		self.weight = nn.Parameter(torch.ones(dim))

	def _norm(self, x: torch.Tensor):
		# (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
		# rsqrt: 1 / sqrt(x)
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x: torch.Tensor):
		# (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
		return self.weight * self._norm(x.float()).type_as(x)


class RMSNormGT1(nn.Module):
	"""This Ground Truth implementation was provided with the exercise."""
	def __init__(self, dim: int, eps: float = 1e-6):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(dim))
		self.variance_epsilon = eps

	def forward(self, hidden_states):
		input_dtype = hidden_states.dtype
		hidden_states = hidden_states.to(torch.float32)
		variance = hidden_states.pow(2).mean(-1, keepdim=True)
		hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
		return self.weight * hidden_states.to(input_dtype)


class RMSNormL3(torch.nn.Module):
	"""Implementation extracted from LLAMA3."""
	def __init__(self, dim: int, eps: float = 1e-6):
		super().__init__()
		self.eps = eps
		self.weight = nn.Parameter(torch.ones(dim))

	def _norm(self, x):
		return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

	def forward(self, x):
		output = self._norm(x.float()).type_as(x)
		return output * self.weight	
