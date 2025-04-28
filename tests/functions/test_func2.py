import pytest
import torch
from optHIM.functions.func2 import Func2


class TestFunc2:
    """Test suite for the func2 function."""
    
    @pytest.fixture
    def setup_func2(self):
        """Setup the func2 function for testing."""
        return Func2()
    
    def test_function_value(self, setup_func2):
        """Test the function value computation."""
        function = setup_func2
        
        X = torch.tensor([[0.0, 0.0], 
                          [1.0, 1.0], 
                          [1.0, -1.0], 
                          [-1.0, 1.0], 
                          [-1.0, -1.0]], dtype=torch.float32)
        #  f(z1, z2) = (exp(z1) - 1)/(exp(z1) + 1) + 0.1*exp(-z1) + sum_{i=2...n} (zi - 1)^4
        Y_gt = torch.tensor([14.203125, 14.203125, 5.703125, 14.203125, 38.703125], dtype=torch.float32)
        for x, y_gt in zip(X, Y_gt):
            y = function(x)
            assert torch.allclose(y_gt, y, atol=1e-3), \
                f"expected {y_gt}, got {y}"

    def test_gradient(self, setup_func2):
        """Test the gradient computation."""
        function = setup_func2
        
        X = torch.tensor([[0.0, 0.0], 
                          [1.0, 1.0], 
                          [1.0, -1.0], 
                          [-1.0, 1.0], 
                          [-1.0, -1.0]], dtype=torch.float32, requires_grad=True)
        grad_X_gt = torch.tensor([[-12.75, 0.0], 
                                  [0.0, 27.75], 
                                  [-0.5, -6.25], 
                                  [0.0, -27.75], 
                                  [-32.5, -25.75]], dtype=torch.float32)
        for x, grad_x_gt in zip(X, grad_X_gt):
            y = function(x)
            grad_x = torch.autograd.grad(y, x)[0]
            assert torch.allclose(grad_x_gt, grad_x, atol=1e-3), \
                f"expected {grad_x_gt}, got {grad_x}"