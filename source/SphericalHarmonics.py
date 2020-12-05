import numpy as np

_ONE_OVER_PI = 1.0 / np.pi
_ONE_OVER_SQRT_PI = 1.0 / np.sqrt(np.pi)
_SQRT_3 = np.sqrt(3)
_SQRT_5 = np.sqrt(5)


def normalization_coefficient(order, degree):
    assert (order >= 0 and -order <= degree <= order)
    return np.power(-1.0, degree) * np.math.factorial(order - degree) / np.math.factorial(order + degree)


def get_sh_basis_value_cartesian(x, y, z, order, degree, normalize=False):
    """
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param order: also named l
    :param degree:  also named m
    :param normalize: use normalized basis function
    :return: sh basis function value
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

    assert(order >= 0 and -order <= degree <= order)

    if order == 0:
        result = np.full_like(x, 0.5 * _ONE_OVER_SQRT_PI)
    elif order == 1:
        coefficient = 0.5 * _SQRT_3 * _ONE_OVER_SQRT_PI
        if degree == -1:
            result = y * coefficient
        elif degree == 0:
            result = z * coefficient
        else:
            result = x * coefficient
    elif order == 2:
        coefficient = 0.5 * _SQRT_5 * _ONE_OVER_SQRT_PI
        if degree == -2:
            result = coefficient * _SQRT_3 * x * y
        elif degree == -1:
            result = coefficient * _SQRT_3 * y * z
        elif degree == 0:
            result = 0.5 * coefficient * (3 * (z * z) - 1)
        elif degree == 1:
            result = coefficient * _SQRT_3 * z * x
        else:
            result = 0.5 * coefficient * _SQRT_3 * (x * x - y * y)
    else:
        result = np.full_like(x, 0)

    if normalize:
        result *= normalization_coefficient(order, degree)

    return result
