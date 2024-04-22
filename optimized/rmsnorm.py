import torch
import torch.nn as nn

import triton
import triton.language as tl

@triton.jit
def square_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
	# The rows are independent, so we parallelize across those
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

@triton.jit
def mean_of_squares_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
	# The rows are independent, so we parallelize across those
	row_idx = tl.program_id(0)
	# The stride represents how much we need to increase the pointer to advance 1 row
	row_start_ptr = input_ptr + row_idx * input_row_stride
	# The block size is the next power of two greater than n_cols, so we can fit each
	# row in a single block
	col_offsets = tl.arange(0, BLOCK_SIZE)
	input_ptrs = row_start_ptr + col_offsets
	# Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
	row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
	tl.debug_barrier()

	square_output = row * row
	mean_output = tl.sum(square_output)/n_cols + eps
	
	# Write back output to DRAM
	output_row_start_ptr = output_ptr + row_idx * output_row_stride # TODO: optimization: always 1 after the reduction
	output_ptrs = output_row_start_ptr + col_offsets
	tl.store(output_ptrs, mean_output, mask=col_offsets < n_cols)

@triton.jit
def rms_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, eps, BLOCK_SIZE: tl.constexpr):
	# The rows are independent, so we parallelize across those
	row_idx = tl.program_id(0)
	# The stride represents how much we need to increase the pointer to advance 1 row
	row_start_ptr = input_ptr + row_idx * input_row_stride
	# The block size is the next power of two greater than n_cols, so we can fit each
	# row in a single block
	col_offsets = tl.arange(0, BLOCK_SIZE)
	input_ptrs = row_start_ptr + col_offsets
	# Load the row into SRAM, using a mask since BLOCK_SIZE may be > than n_cols
	row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))
	tl.debug_barrier()

	square_output = row * row
	rms = tl.sqrt(tl.sum(square_output)/n_cols + eps)
	
	# Write back output to DRAM
	output_row_start_ptr = output_ptr + row_idx * output_row_stride # TODO: optimization: always 1 after the reduction
	output_ptrs = output_row_start_ptr + col_offsets
	tl.store(output_ptrs, rms, mask=col_offsets < n_cols)


@triton.jit
def rms_norm(output_ptr, input_ptr, weights_ptr, stride, N, eps, DTYPE:tl.constexpr, BLOCK_SIZE: tl.constexpr):
    """
    RMS Norm Triton Kernel

    Params:
        - input_ptr (tensor): Pointer to Input
        - output_ptr (tensor): Pointer to Output
        - weights_ptr (tensor): Pointer to Scale applied to the normalized input
        - stride (int): Stride to be applied when accessing elements in the input and output tensors
        - N (int): Number of elements to be reduced == input_ptr.shape[-1]
        - eps (half/float): Epsilon value added to the variance to prevent division by zero
        - BLOCK_SIZE (constexpr): Size of the block for computation, provided as a compile-time constant

    Usage:
        _rms_norm[grid, block](x, y, self.w, input_stride , N, eps, BLOCK_SIZE)
    """
    row = tl.program_id(0)
    output_ptr += row * stride
    input_ptr += row * stride

    tmp = 0
    tmp = tl.zeros([BLOCK_SIZE], dtype=DTYPE)
    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        a = tl.load(input_ptr + cols, mask=mask, other=0.0).to(DTYPE)
        tmp += a * a
    rms = tl.sqrt(tl.sum(tmp) / N + eps)

    for offset in range(0, N, BLOCK_SIZE):
        cols = offset + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(input_ptr + cols, mask=mask, other=0.0, eviction_policy="evict_first").to(DTYPE)
        w = tl.load(weights_ptr + cols, mask=mask)
        x_hat = x / rms
        y = x_hat * w
        tl.store(output_ptr + cols, y, mask=mask)


class RMSNormTriton(nn.Module):
	def __init__(self, dim: int, eps: float = 1e-6):
		super().__init__()
		self.eps = eps
		# The gamma parameter
		self.weight = nn.Parameter(torch.ones(dim))


	def forward(self, x: torch.Tensor):
		"""Generates BxSeqLen tensors of innermost dimension ModelDim."""
		# (ModelDim) * (B, SeqLen, ModelDim) = (B, SeqLen, ModelDim)
  
		# For use with _square, _mean_of_squares, _rms functions
		# return self.weight * self._norm(x.float()).type_as(x)
  
		# For use with _rms_norm function
		return self._rms_norm(x, self.weight, self.eps).type_as(x)
	

	# def _norm(self, x: torch.Tensor):
	# 	# SeqLen represents the number of tokens in the sequence
	# 	# (B, SeqLen, ModelDim) * (B, SeqLen, 1) = (B, SeqLen, ModelDim)
	# 	# return x * torch.rsqrt(self._square(x).mean(-1, keepdim=True) + self.eps) # 2.07ms @ 4096x4096
	# 	# return x * torch.rsqrt(self._mean_of_squares(x) + self.eps) # 1.30ms @ 4096x4096
	# 	return 1.0 / self._rms(x, self.eps) # 1.21ms @ 4096x4096
	
	def _square(self, x):
		"""Square the input tensor element-wise."""

		# Flatten the tensor except for the last dimension
		x_reshaped = x.reshape(-1, x.shape[-1])
		n_rows, n_cols = x_reshaped.shape
		# The block size is the smallest power of two greater than the number of columns in `x`
		BLOCK_SIZE = triton.next_power_of_2(n_cols)
		# Another trick we can use is to ask the compiler to use more threads per row by
		# increasing the number of warps (`num_warps`) over which each row is distributed.
		
		# Allocate output
		y = torch.empty_like(x_reshaped)
		# Enqueue kernel. The 1D launch grid is simple: we have one kernel instance per row of the input matrix
		square_kernel[(n_rows, )](
			y,
			x_reshaped,
			x_reshaped.stride(0),
			y.stride(0),
			n_cols,
			BLOCK_SIZE=BLOCK_SIZE,
		)
		# Reshape the output tensor to match the original shape of x
		y = y.reshape(*x.shape)
		return y
	
	def _mean_of_squares(self, x, eps=1e-6):
		"""
		Compute the mean of the squares of the input tensor.
		
		Params:
			- x (torch.Tensor): The input tensor of shape (B, SeqLen, ModelDim)
			- eps (float): A small value to add to the denominator for numerical stability
		
		Returns:
			- y (torch.Tensor): The mean of the squares of the input tensor of shape (B, SeqLen, 1)
		"""
		# Flatten the tensor except for the last dimension
		x_reshaped = x.reshape(-1, x.shape[-1])
		n_rows, n_cols = x_reshaped.shape
		# The block size is the smallest power of two greater than the number of columns in `x`
		BLOCK_SIZE = triton.next_power_of_2(n_cols)

		# Allocate output
		# Get the shape of x_reshaped and replace the last dimension with 1
		# This dimension will be reduced with the calculation of the mean
		new_shape = (*x_reshaped.shape[:-1], 1)

		# Create a new tensor with the new shape
		y = torch.empty(new_shape, device=x_reshaped.device, dtype=x_reshaped.dtype)
  
		mean_of_squares_kernel[(n_rows, )](
			y,
			x_reshaped,
			x_reshaped.stride(0),
			y.stride(0), # TODO: optimization: always 1 after the reduction
			n_cols,
			eps,
			BLOCK_SIZE=BLOCK_SIZE,
		)
		# Reshape the output tensor, we reduced the last dimension to 1
		y = y.reshape(*x.shape[:-1], 1)
		return y
	
	def _rms(self, x, eps=1e-6):
		"""
		Compute the square root of the mean of the squares of the input tensor.
		
		Params:
			- x (torch.Tensor): The input tensor of shape (B, SeqLen, ModelDim)
			- eps (float): A small value to add to the denominator for numerical stability
		
		Returns:
			- y (torch.Tensor): The rms of the input tensor of shape (B, SeqLen, 1)
		"""

		# Flatten the tensor except for the last dimension
		x_reshaped = x.reshape(-1, x.shape[-1])
		n_rows, n_cols = x_reshaped.shape
		# The block size is the smallest power of two greater than the number of columns in `x`
		BLOCK_SIZE = triton.next_power_of_2(n_cols)
		
		# Allocate output
		# Get the shape of x_reshaped and replace the last dimension with 1
		# This dimension will be reduced with the calculation of the mean
		new_shape = (*x_reshaped.shape[:-1], 1)

		# Create a new tensor with the new shape
		y = torch.empty(new_shape, device=x_reshaped.device, dtype=x_reshaped.dtype)
  
		mean_of_squares_kernel[(n_rows, )](
			y,
			x_reshaped,
			x_reshaped.stride(0),
			y.stride(0), # TODO: optimization: always 1 after the reduction
			n_cols,
			eps,
			BLOCK_SIZE=BLOCK_SIZE,
		)
		# Reshape the output tensor, we reduced the last dimension to 1
		y = y.reshape(*x.shape[:-1], 1)
		return y
	
	def _rms_norm(self, x, w, eps):
		"""
		Compute the RMS normalization of the input tensor.

		Params:
			- x (torch.Tensor): The input tensor of shape (B, SeqLen, ModelDim)
			- w (torch.Tensor): The gamma parameter of shape (ModelDim)
			- eps (half/float): A small value to add to the denominator for numerical stability

		Returns:
			- y (torch.Tensor): The normalized tensor of shape (B, SeqLen, ModelDim)
		"""
		# Flatten the tensor except for the last dimension
		x_reshaped = x.reshape(-1, x.shape[-1])
		n_rows, n_cols = x_reshaped.shape
		M= n_rows
		N= n_cols
		# The block size is the smallest power of two greater than the number of columns in `x`
		BLOCK_SIZE = triton.next_power_of_2(n_cols)

		data_type = x.dtype
		if data_type == torch.float32:
			data_type = tl.float32
		elif data_type == torch.float16:
			data_type = tl.float16
		elif data_type == torch.int64:
			data_type = tl.int64
		else:
			raise ValueError(f"Unsupported data type: {data_type}")
		
		# Allocate output
		y = torch.empty_like(x_reshaped)
		rms_norm[(M,)](
			y,
			x_reshaped, 
			w,
			x_reshaped.stride(0), N, eps,
			DTYPE=data_type,
			BLOCK_SIZE=BLOCK_SIZE
		)
        
		# Reshape the output tensor
		y = y.reshape(*x.shape)
		return y	
	