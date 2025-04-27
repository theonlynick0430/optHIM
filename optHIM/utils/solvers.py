from math import sqrt


def quadratic_2D(a, b, c):
    """
    Solve a quadratic equation of the form ax^2 + bx + c = 0.

    Args:
        a (float): coefficient of x^2
        b (float): coefficient of x
        c (float): constant term

    Returns:
        x_soln (tuple): solutions to the quadratic equation
    """
    determinant = b**2 - 4*a*c
    if determinant < 0:
        raise ValueError("No real solutions")
    x1 = (-b + sqrt(determinant)) / (2*a)
    x2 = (-b - sqrt(determinant)) / (2*a)
    return x1, x2