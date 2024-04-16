This implementation was extracted from https://github.com/leimao/Nsight-Compute-Docker-Image/blob/main/pytorch_square.py which did not provide a license.

Notes, `torch.square()` and `a**2` seems to use `aten:pow`. `a*a` uses `aten:square()`.

```
=============
Profiling torch.square
=============
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
             aten::square         1.67%      17.000us        12.07%     123.000us     123.000us      15.000us         1.39%       1.079ms       1.079ms             1  
                aten::pow         7.07%      72.000us         9.91%     101.000us     101.000us       1.054ms        97.68%       1.064ms       1.064ms             1  
        aten::result_type         0.10%       1.000us         0.10%       1.000us       1.000us       6.000us         0.56%       6.000us       6.000us             1  
                 aten::to         0.00%       0.000us         0.00%       0.000us       0.000us       4.000us         0.37%       4.000us       4.000us             1  
          cudaEventRecord         2.36%      24.000us         2.36%      24.000us       3.000us       0.000us         0.00%       0.000us       0.000us             8  
         cudaLaunchKernel         1.96%      20.000us         1.96%      20.000us      20.000us       0.000us         0.00%       0.000us       0.000us             1  
    cudaDeviceSynchronize        86.85%     885.000us        86.85%     885.000us     885.000us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.019ms
Self CUDA time total: 1.079ms

=============
Profiling a * a
=============
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                aten::mul         2.99%      29.000us         4.33%      42.000us      42.000us       1.004ms       100.00%       1.004ms       1.004ms             1  
          cudaEventRecord         0.93%       9.000us         0.93%       9.000us       4.500us       0.000us         0.00%       0.000us       0.000us             2  
         cudaLaunchKernel         1.34%      13.000us         1.34%      13.000us      13.000us       0.000us         0.00%       0.000us       0.000us             1  
    cudaDeviceSynchronize        94.74%     919.000us        94.74%     919.000us     919.000us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 970.000us
Self CUDA time total: 1.004ms

=============
Profiling a ** 2
=============
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                     Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
                aten::pow         7.70%      78.000us        11.25%     114.000us     114.000us       1.056ms        98.42%       1.073ms       1.073ms             1  
        aten::result_type         0.10%       1.000us         0.10%       1.000us       1.000us       9.000us         0.84%       9.000us       9.000us             1  
                 aten::to         0.00%       0.000us         0.00%       0.000us       0.000us       8.000us         0.75%       8.000us       8.000us             1  
          cudaEventRecord         3.06%      31.000us         3.06%      31.000us       5.167us       0.000us         0.00%       0.000us       0.000us             6  
         cudaLaunchKernel         2.17%      22.000us         2.17%      22.000us      22.000us       0.000us         0.00%       0.000us       0.000us             1  
    cudaDeviceSynchronize        86.97%     881.000us        86.97%     881.000us     881.000us       0.000us         0.00%       0.000us       0.000us             1  
-------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  
Self CPU time total: 1.013ms
Self CUDA time total: 1.073ms
```
