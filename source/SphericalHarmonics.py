import numpy as np
from Math import spherical_to_cartesian

_ONE_OVER_PI = 1.0 / np.pi
_ONE_OVER_SQRT_PI = 1.0 / np.sqrt(np.pi)
_SQRT_2 = np.sqrt(2)
_SQRT_3 = np.sqrt(3)
_SQRT_5 = np.sqrt(5)
_SQRT_7 = np.sqrt(7)
_SQRT_15 = np.sqrt(15)


_SH_0 = [0.282094791]
_SH_1 = [0.488602511, 0.488602511, 0.488602511]
_SH_2 = [1.092548431, 1.092548431, 0.315391565, 1.092548431, 0.546274215]


def get_sh_basis_value_radius_in_cartesian(phi, theta, degree, order):
    phi, theta = np.asarray(phi), np.asarray(theta)
    x, y, z = spherical_to_cartesian(phi, theta, 1.0)
    r = get_sh_basis_value_cartesian(x, y, z, degree, order)
    color = r > 0
    color = color.astype(np.float)
    color = 2.0 * color - 1.0
    x, y, z = spherical_to_cartesian(phi, theta, np.abs(r))

    return x, y, z, color


def get_sh_basis_value_cartesian(x, y, z, degree, order):
    """
    :param x: x coordinate
    :param y: y coordinate
    :param z: z coordinate
    :param degree: also named l
    :param order:  also named m
    :return: sh basis function value
    """
    x, y, z = np.asarray(x), np.asarray(y), np.asarray(z)

    assert(degree >= 0 and -degree <= order <= degree)

    if degree == 0:
        result = np.full_like(x, 0.5 * _ONE_OVER_SQRT_PI)

    elif degree == 1:
        coefficient = 0.5 * _SQRT_3 * _ONE_OVER_SQRT_PI
        if order == -1:
            result = y * coefficient
        elif order == 0:
            result = z * coefficient
        else:
            result = x * coefficient

    elif degree == 2:
        coefficient = 0.5 * _SQRT_5 * _ONE_OVER_SQRT_PI
        if order == -2:
            result = coefficient * _SQRT_3 * x * y
        elif order == -1:
            result = coefficient * _SQRT_3 * y * z
        elif order == 0:
            result = 0.5 * coefficient * (3 * (z * z) - 1)
        elif order == 1:
            result = coefficient * _SQRT_3 * z * x
        else:
            result = 0.5 * coefficient * _SQRT_3 * (x * x - y * y)

    elif degree == 3:
        coefficient = 0.25 * _SQRT_7 * _ONE_OVER_SQRT_PI
        if order == -3:
            result = coefficient * (_SQRT_5 / _SQRT_2) * y * (3 * x * x - y * y)
        elif order == -2:
            result = coefficient * 2.0 * _SQRT_15 * x * y * z
        elif order == -1:
            result = coefficient * (_SQRT_3 / _SQRT_2) * y * (4 * z * z - x * x - y * y)
        elif order == 0:
            result = coefficient * (z * (2 * z * z - 3 * x * x - 3 * y * y))
        elif order == 1:
            result = coefficient * (_SQRT_3 / _SQRT_2) * x * (4 * z * z - x * x - y * y)
        elif order == 2:
            result = coefficient * _SQRT_15 * z * (x * x - y * y)
        else:
            result = coefficient * (_SQRT_5 / _SQRT_2) * x * (x * x - 3 * y * y)

    else:
        result = np.full_like(x, 0)

    return result


def compute_sh_coefficient_from_image(image, degree, order):
    height = image.shape[0]
    width = image.shape[1]
    phi, theta = np.mgrid[0:2 * np.pi:complex(0, width), 0:np.pi:complex(0, height)]
    x, y, z = spherical_to_cartesian(phi, theta, 1.0)

    sin_theta = np.sin(theta)
    sin_theta = sin_theta[:, :, np.newaxis]
    sin_theta = np.concatenate((sin_theta, sin_theta, sin_theta), axis=2)

    sh_basis = get_sh_basis_value_cartesian(x, y, z, degree, order)
    sh_basis = sh_basis[:, :, np.newaxis]
    sh_basis = np.concatenate((sh_basis, sh_basis, sh_basis), axis=2)

    input_light = np.transpose(image, [1, 0, 2]) / 255.0

    # Riemann sum, area is sin(theta) * (2pi / width) * (pi / height)
    delta_area = (2 * np.pi) * np.pi / (height * width)
    coefficients = np.sum(np.multiply(input_light, sh_basis) * sin_theta, axis=(0, 1)) * delta_area

    output_image = np.zeros((width, height, 3))

    for i in range(0, 3):
        output_image[:, :, i] = sh_basis[:, :, 1] * coefficients[i]

    return output_image
