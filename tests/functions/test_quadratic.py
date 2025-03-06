import pytest
import torch
from optim.functions.quadratic import Quadratic


class TestQuadratic:
    """Test suite for the quadratic model."""
    
    @pytest.fixture
    def setup_quadratic(self):
        """Setup a simple quadratic model for testing."""
        A = torch.tensor([[4.0, 1.0], [1.0, 2.0]], dtype=torch.float32)
        b = torch.tensor([1.0, -1.0], dtype=torch.float32)
        c = torch.tensor(2.0, dtype=torch.float32)
        return Quadratic(A, b, c)
    
    def test_function_value(self, setup_quadratic):
        """Test the function value computation."""
        model = setup_quadratic
        
        X = torch.tensor([[0.0, 0.0], 
                          [1.0, 1.0], 
                          [1.0, -1.0], 
                          [-1.0, 1.0], 
                          [-1.0, -1.0]], dtype=torch.float32)
        # f(x, y) = 2x^2 + y^2 + xy + x - y + 2 
        Y_gt = torch.tensor([2.0, 6.0, 6.0, 2.0, 6.0], dtype=torch.float32)
        for x, y_gt in zip(X, Y_gt):
            y = model(x)
            assert torch.allclose(y_gt, y), \
                f"expected {y_gt}, got {y}"

    def test_gradient(self, setup_quadratic):
        """Test the gradient computation."""
        model = setup_quadratic

        X = torch.tensor([[0.0, 0.0], 
                          [1.0, 1.0], 
                          [1.0, -1.0], 
                          [-1.0, 1.0], 
                          [-1.0, -1.0]], dtype=torch.float32, requires_grad=True)
        # grad f(x, y) = [4x + y + 1, 2y + x - 1]
        grad_X_gt = torch.tensor([[1.0, -1.0], 
                                  [6.0, 2.0], 
                                  [4.0, -2.0], 
                                  [-2.0, 0.0], 
                                  [-4.0, -4.0]], dtype=torch.float32)
        for x, grad_x_gt in zip(X, grad_X_gt):
            y = model(x)
            grad_x = torch.autograd.grad(y, x)[0]
            assert torch.allclose(grad_x_gt, grad_x), \
                f"expected {grad_x_gt}, got {grad_x}"