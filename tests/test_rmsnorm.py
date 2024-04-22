import torch
import unittest
from baselines.rmsnorm import MyRMSNorm, RMSNormGT1, RMSNormL3, RMSNormL3Half
from optimized.rmsnorm import RMSNormTriton

class TestRMSNorm(unittest.TestCase):
    def test_forward_my(self):
        
        x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        
        # Compute the output
        rmsnorm_test = MyRMSNorm(dim=3)
        expected_output = rmsnorm_test.forward(x)
        
        # Compute the actual output
        rmsnorm_correct = RMSNormL3(dim=3)
        actual_output = rmsnorm_correct.forward(x)
        
        self.assertTrue(torch.all(torch.eq(actual_output, expected_output)))

    def test_forward_gt1(self):
        
        x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        
        # Compute the output
        rmsnorm_test = RMSNormGT1(dim=3)
        expected_output = rmsnorm_test.forward(x)
        
        # Compute the actual output
        rmsnorm_correct = RMSNormL3(dim=3)
        actual_output = rmsnorm_correct.forward(x)
        
        self.assertTrue(torch.all(torch.eq(actual_output, expected_output)))

    def test_forward_triton(self):
        
        x = torch.tensor([[[1, 2, 3], [5, 5, 5]], [[7, 8, 9], [10, 11, 12]]])
        # Change all to float
        x = x.float()
        
        # Compute the actual output
        rmsnorm_correct = RMSNormL3(dim=3)
        actual_output = rmsnorm_correct.forward(x)
        
        # Compute the output
        rmsnorm_test = RMSNormTriton(dim=3).cuda()
        expected_output = rmsnorm_test.forward(x.to('cuda:0'))
        
        self.assertTrue(torch.allclose(actual_output, expected_output.to('cpu')))
    
    def test_forward_triton_half(self):
        
        x = torch.tensor([[[1, 2, 3], [5, 5, 5]], [[7, 8, 9], [10, 11, 12]]])
        # Change all to float
        x = x.half()
        
        # Compute the actual output
        rmsnorm_correct = RMSNormL3Half(dim=3).half()
        actual_output = rmsnorm_correct.forward(x)
        
        # Compute the output
        rmsnorm_test = RMSNormTriton(dim=3).cuda()
        expected_output = rmsnorm_test.forward(x.to('cuda:0'))

        # TODO: It appears that operating in half precision on CPU and GPU is not the same as in float precision 
        #       Absolute tolerance was relaxed from 1e-8 to 1e-3
        self.assertTrue(torch.allclose(actual_output.to('cpu'), expected_output.to('cpu'), atol=1e-3))

if __name__ == '__main__':
    unittest.main()