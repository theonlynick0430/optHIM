import torch
import pytest
from optim.functions.rosenbrock import Rosenbrock


class TestRosenbrock:
    """Test suite for the rosenbrock model."""
    
    @pytest.fixture
    def setup_rosenbrock(self):
        """Setup the rosenbrock function for testing."""
        return Rosenbrock()
    
    def test_function_value(self, setup_rosenbrock):
        """Test the function value computation."""
        model = setup_rosenbrock
        
        X = torch.tensor([[0.0, 0.0], 
                          [1.0, 1.0], 
                          [1.0, -1.0], 
                          [-1.0, 1.0], 
                          [-1.0, -1.0]], dtype=torch.float32)
        # f(x, y) = (1-x)^2 + 100*(y-x^2)^2
        Y_gt = torch.tensor([1.0, 0.0, 400.0, 4.0, 404.0], dtype=torch.float32)
        for x, y_gt in zip(X, Y_gt):
            y = model(x)
            assert torch.allclose(y_gt, y), \
                f"expected {y_gt}, got {y}"

    def test_gradient(self, setup_rosenbrock):
        """Test the gradient computation."""
        model = setup_rosenbrock
        
        X = torch.tensor([[0.0, 0.0], 
                          [1.0, 1.0], 
                          [1.0, -1.0], 
                          [-1.0, 1.0], 
                          [-1.0, -1.0]], dtype=torch.float32, requires_grad=True)
        # grad f(x, y) = [-2 * (1 - x) - 400 * (y - x ** 2) * x, 200 * (y - x ** 2)]
        grad_X_gt = torch.tensor([[-2.0, 0.0], 
                                  [0.0, 0.0], 
                                  [800.0, -400.0], 
                                  [-4.0, 0.0], 
                                  [-804.0, -400.0]], dtype=torch.float32)
        for x, grad_x_gt in zip(X, grad_X_gt):
            y = model(x)
            grad_x = torch.autograd.grad(y, x)[0]
            assert torch.allclose(grad_x_gt, grad_x), \
                f"expected {grad_x_gt}, got {grad_x}"