import torch
import pytest
from optim.functions.rosenbrock import Rosenbrock


class TestRosenbrock:
    """Test suite for the rosenbrock model."""
    
    @pytest.fixture
    def setup_rosenbrock(self):
        """Setup the rosenbrock function for testing."""
        # create model
        model = Rosenbrock()
        
        # known values for testing
        x_star = torch.ones((2, 1), dtype=torch.float32)  # global minimum at (1, 1)
        f_star = torch.tensor(0.0, dtype=torch.float32)  # function value at minimum
        
        return {
            'model': model,
            'x_star': x_star,
            'f_star': f_star
        }
    
    def test_function_value(self, setup_rosenbrock):
        """Test the function value computation."""
        data = setup_rosenbrock
        model = data['model']
        
        # test at x = [0, 0]
        x = torch.zeros((2, 1), dtype=torch.float32)
        # f([0,0]) = (1-0)^2 + 100*(0-0^2)^2 = 1
        expected_value = torch.tensor(1.0, dtype=torch.float32)
        computed_value = model(x)
        assert torch.isclose(expected_value, computed_value), \
            f"expected {expected_value}, got {computed_value}"
        
        # test at minimum point
        x_star = data['x_star']
        expected_value = data['f_star']
        computed_value = model(x_star)
        assert torch.isclose(expected_value, computed_value), \
            f"expected {expected_value}, got {computed_value}"
    
    def test_gradient(self, setup_rosenbrock):
        """Test the gradient computation."""
        data = setup_rosenbrock
        model = data['model']
        
        # test at x = [0, 0]
        x = torch.zeros((2, 1), dtype=torch.float32, requires_grad=True)
        y = model(x)
        expected_grad = torch.tensor([[-2.0], [0.0]], dtype=torch.float32)
        computed_grad = torch.autograd.grad(y, x)[0]
        assert torch.allclose(expected_grad, computed_grad), \
            f"expected {expected_grad}, got {computed_grad}"
        
        # test at minimum point
        x_star = data['x_star'].clone()
        x_star.requires_grad_(True)
        y = model(x_star)
        computed_grad = torch.autograd.grad(y, x_star)[0]
        assert torch.allclose(torch.zeros_like(computed_grad), computed_grad, atol=1e-5), \
            f"expected zero gradient at minimum, got {computed_grad}" 