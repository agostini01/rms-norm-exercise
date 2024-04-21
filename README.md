# Implementation of RMS Normalization

Root Mean Square (RMS) normalization is a technique used to normalize the
amplitude of a signal. It is used in many signal processing applications, such
as audio processing, image processing, and data analysis.  It is also a layer in the `LLama2` model, which is a state-of-the-art model for machine translation.


In the [paper](https://dl.acm.org/doi/pdf/10.5555/3454287.3455397) "Root Mean Square Layer Normalization" by Zhang and Sennrich, the authors hypothesize that the re-scaling invariance is the reason for success of LayerNorm, rather than re-centering invariance. As such, they propose a new normalization technique called Root Mean Square Layer Normalization (RMSNorm) that normalizes the input activations to have unit root mean square (RMS) value.

```
@article{zhang2019root,
  title={Root mean square layer normalization},
  author={Zhang, Biao and Sennrich, Rico},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}
```


## Formula

$$\bar{a_i} = \frac{a_i}{RMS(\vec{a})}g_i$$

where:

$$RMS(\vec{a}) = \sqrt{eps + \frac{1}{n} \sum_{i=1}^{n} a_i^2}$$

* $\bar{a_i}$ (a-bar-i): This represents the **normalized value** of element $i$ after applying RMSNorm. The normalization process rescales the values in $a$ to have a different scale compared to the original values.
* $a_i$: This represents the **original value** of element $i$ in the input vector $a$ before any normalization is applied.
* $RMS(a)$: This term represents the **Root Mean Square** of the elements in vector $a$. It calculates a single value that indicates the spread or magnitude of the values in $a$. 
* $g_i$ (gamma-i): This represents a **learnable scaling factor**, specific to element $i$. It's not included in all formulations of RMSNorm. Some implementations allow the model to learn this factor to introduce more control over the normalization process for each element.
* $eps$: This is a small value (usually close to zero) added to the denominator to prevent division by zero. It's a common practice to avoid numerical instability when calculating the RMS value.


## Torch Implementation

```python
class RMSNorm(nn.Module):
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
```


## Use in LLama

![LLama2 Architecture](res/llama-arch.png)
Image from: https://github.com/hkproj/pytorch-llama/blob/main/Slides.pdf

## Assignment

- Implement RMSNorm in CUDA, for 3D tensors using half precision
- Document thought process
- Identify and discuss potential optimizations
- Model size refered to as Dim is `>=4096` and multiple of `32`

## Thought Process

0. Setup some profiling infrastructure
    - `torch.profiler` and `nsys` for "end-to-end"
    - `ncu` for kernel evaluation
    - I will start by keeping `Dim` constant and varying `SeqLen`, with `Batch Size of 1`
1. Study and Implement RMSNorm in python to serve as a baseline
2. Since we have 3D tensors as inputs, I assume the dimensions (B, SeqLen, Dim)
    - `B` represents number of batches
        - All batches are assumed to have the **same** size of **(SeqLen, Dim)**
    - `SeqLen` represents the size of the sequence, i.e. tokens in a batch
    - `Dim` represents the size of the embeddings to represent a token
        - Also represents the size of the trainable parameter (weights) vector $\bar{g}$
3. All inputs are computed against the same $\bar{g}$ weights
        - The same set of learnable scaling factors ($g_i$) would be applied to all input vectors in the sequence
4. It makes sense to keep the $\bar{g}$ in GPU Memmory, and reuse the values until we go over all (B, SeqLen, : ) examples
5. The RMSNorm has many subkernels, some depend on the weights. Fusion can be relevant so we can re-use intermediate results when possible.
6. I will start by implementing each kernel individually, from innermost to outermost
    1. square
    2. mean (reduction('+') then devide by N)
    3. add single scalar (eps)
    4. sqrt
    5. inverse
    6. element_wise multiply
7. Many of these kernels can reuse builtin libraries from [CUB](https://nvidia.github.io/cccl/cub/modules.html). We can also implement these kernels with [Triton](https://github.com/openai/triton)

## Additional/Potential Optimizations

- Fusion of kernels
- Keeping weights in shared memory
- Reduction can leverage a parallel implementation


## Additional Resources

LLamma2 implementation from scratch:
- https://www.youtube.com/watch?v=oM4VmoabDAI&ab_channel=UmarJamil

Integrating kernel in pytorch:
- https://github.com/cuda-mode/profiling-cuda-in-torch/tree/main/load_inline_cuda
