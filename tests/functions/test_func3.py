import pytest
import torch
from optim.functions.func3 import Func3


class TestFunc3:
    """Test suite for the func3 model."""
    
    @pytest.fixture
    def setup_func3(self):
        """Setup the func3 function for testing."""
        return Func3()
    
    def test_function_value(self, setup_func3):
        """Test the function value computation."""
        model = setup_func3
        
        X = torch.tensor([[0.0, 0.0], 
                          [1.0, 1.0], 
                          [1.0, -1.0], 
                          [-1.0, 1.0], 
                          [-1.0, -1.0]], dtype=torch.float32)
        # f(w, z) = (1.5-w(1-z))^2 + (2.25-w(1-z^2))^2 + (2.625-w(1-z^3))^2
        Y_gt = torch.tensor([1.1, 0.49890510137, 16.4989051014, -0.19028897441, 15.8097110256], dtype=torch.float32)
        for x, y_gt in zip(X, Y_gt):
            y = model(x)
            assert torch.allclose(y_gt, y, atol=1e-3), \
                f"expected {y_gt}, got {y}"

    def test_gradient(self, setup_func3):
        """Test the gradient computation."""
        model = setup_func3
        
        X = torch.tensor([[0.0, 0.0], 
                          [1.0, 1.0], 
                          [1.0, -1.0], 
                          [-1.0, 1.0], 
                          [-1.0, -1.0]], dtype=torch.float32, requires_grad=True)
        grad_X_gt = torch.tensor([[0.4, -4.0], 
                                  [0.3561, 0.0], 
                                  [0.3561, -32.0], 
                                  [0.1212, 0.0], 
                                  [0.1212, -32.0]], dtype=torch.float32)
        for x, grad_x_gt in zip(X, grad_X_gt):
            y = model(x)
            grad_x = torch.autograd.grad(y, x)[0]
            assert torch.allclose(grad_x_gt, grad_x, atol=1e-3), \
                f"expected {grad_x_gt}, got {grad_x}"