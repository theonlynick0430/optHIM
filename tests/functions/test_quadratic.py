import torch
import pytest
from optim.functions.quadratic import Quadratic


class TestQuadratic:
    """Test suite for the quadratic model."""
    
    @pytest.fixture
    def setup_quadratic(self):
        """Setup a simple quadratic model for testing."""
        # positive definite => convex
        A = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float32)
        b = torch.tensor([[1.0], [-1.0]], dtype=torch.float32)
        c = torch.tensor(2.0, dtype=torch.float32)
        
        # analytical minimum at x* = -A^(-1)b
        x_star = -torch.linalg.solve(A, b) # guaranteed to be unique
        # function value at minimum
        f_star = 0.5 * x_star.T @ A @ x_star + b.T @ x_star + c
        f_star = f_star[0, 0]
        
        return {
            'A': A,
            'b': b,
            'c': c,
            'x_star': x_star,
            'f_star': f_star
        }
    
    def test_function_value(self, setup_quadratic):
        """Test the function value computation."""
        data = setup_quadratic
        model = Quadratic(data['A'], data['b'], data['c'])
        
        # test at x = [0, 0]
        x = torch.zeros((2, 1), dtype=torch.float32)
        expected_value = data['c']  # f([0,0]) = c
        computed_value = model(x)
        assert torch.isclose(expected_value, computed_value), \
            f"expected {expected_value}, got {computed_value}"
        
        # test at minimum point
        x_star = data['x_star']
        expected_value = data['f_star']
        computed_value = model(x_star)
        assert torch.isclose(expected_value, computed_value), \
            f"expected {expected_value}, got {computed_value}"
    
    def test_gradient(self, setup_quadratic):
        """Test the gradient computation."""
        data = setup_quadratic
        model = Quadratic(data['A'], data['b'], data['c'])
        
        # test at x = [1, 1]
        x = torch.ones((2, 1), dtype=torch.float32, requires_grad=True)
        y = model(x)
        expected_grad = data['A'] @ x + data['b']
        computed_grad = torch.autograd.grad(y, x)[0]
        assert torch.allclose(expected_grad, computed_grad), \
            f"expected {expected_grad}, got {computed_grad}"
        
        # test at minimum point (gradient should be zero)
        x_star = data['x_star'].clone()
        x_star.requires_grad_(True)
        y = model(x_star)
        computed_grad = torch.autograd.grad(y, x_star)[0]
        assert torch.allclose(torch.zeros_like(computed_grad), computed_grad, atol=1e-5), \
            f"expected zero gradient at minimum, got {computed_grad}" 