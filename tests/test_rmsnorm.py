import torch
import unittest
from baselines.rmsnorm import MyRMSNorm, RMSNormGT1, RMSNormL3

class TestRMSNorm(unittest.TestCase):
    def test_forward_my(self):
        
        x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        
        # Compute the expected output
        rmsnorm_test = MyRMSNorm(dim=3)
        expected_output = rmsnorm_test.forward(x)
        
        # Compute the actual output
        rmsnorm_correct = RMSNormL3(dim=3)
        actual_output = rmsnorm_correct.forward(x)
        
        self.assertTrue(torch.all(torch.eq(actual_output, expected_output)))

    def test_forward_gt1(self):
        
        x = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
        
        # Compute the expected output
        rmsnorm_test = RMSNormGT1(dim=3)
        expected_output = rmsnorm_test.forward(x)
        
        # Compute the actual output
        rmsnorm_correct = RMSNormL3(dim=3)
        actual_output = rmsnorm_correct.forward(x)
        
        self.assertTrue(torch.all(torch.eq(actual_output, expected_output)))

if __name__ == '__main__':
    unittest.main()