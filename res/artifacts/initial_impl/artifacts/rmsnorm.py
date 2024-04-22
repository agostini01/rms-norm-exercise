import torch
import torch.nn as nn

import triton
import triton.language as tl

@triton.jit
def square_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
	# The rows of the softmax are independent, so we parallelize across those
	row_idx = tl.program_id(0)
	# The stride represents how much we need to increase the pointer to advance 1 row
	row_start_ptr = input_ptr + row_idx * input_row_stride
	# The block size is the next power of two greater than n_cols, so we can fit each
	# row in a single block
	col_offsets = tl.arange(0, BLOCK_SIZE)
	input_ptrs = row_start_ptr + col_offsets
	# Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
	row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

	square_output = row * row
	
	# Write back output to DRAM
	output_row_start_ptr = output_ptr + row_idx * output_row_stride
	output_ptrs = output_row_start_ptr + col_offsets
	tl.store(output_ptrs, square_output, mask=col_offsets < n_cols)


class RMSNormTriton(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-6):
		super().__init__()
		self.eps = eps
		# The gamma parameter
		self.weight = nn.Parameter(torch.ones(dim))


	def forward(self, x: torch.Tensor):
		# (Dim) * (B, Seq_Len, Dim) = (B, Seq_Len, Dim)
		return self.weight * self._norm(x.float()).type_as(x)
	

	def _norm(self, x: torch.Tensor):

		# Seq_Len represents the number of tokens in the sequence
		# (B, Seq_Len, Dim) * (B, Seq_Len, 1) = (B, Seq_Len, Dim)
		return x * torch.rsqrt(self._square(x).mean(-1, keepdim=True) + self.eps)
	
	
	def _square(self, x):
		# Flatten the tensor except for the last dimension
		x_reshaped = x.reshape(-1, x.shape[-1])
		n_rows, n_cols = x_reshaped.shape
		# The block size is the smallest power of two greater than the number of columns in `x`
		BLOCK_SIZE = triton.next_power_of_2(n_cols)
		# Another trick we can use is to ask the compiler to use more threads per row by
		# increasing the number of warps (`num_warps`) over which each row is distributed.
		num_warps = 4
		if BLOCK_SIZE >= 2048:
			num_warps = 8
		if BLOCK_SIZE >= 4096:
			num_warps = 16
		# Allocate output
		y = torch.empty_like(x_reshaped)
		# Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row of the input matrix
		square_kernel[(n_rows, )](
			y,
			x_reshaped,
			x_reshaped.stride(0),
			y.stride(0),
			n_cols,
			num_warps=num_warps,
			BLOCK_SIZE=BLOCK_SIZE,
		)
		# Reshape the output tensor to match the original shape of x
		y = y.reshape(*x.shape)
		return y