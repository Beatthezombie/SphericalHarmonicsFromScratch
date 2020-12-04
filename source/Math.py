import numpy as np


def generate_sphere_coordinates(radius=1.0, samples=25):
    """
    Generate spherical coordinates using a grid of angle values, the convention used is:
    r in (0, infinity)
    theta in (0, pi), rotating from the polar axis (z) (polar angle)
    phi in (0, 2pi), rotates around the z axis
    :param radius: radius of sphere
    :param samples: number of samples to use for each angle, creates a grid of sample * (2 * samples)
    :return: x,y,z arrays
    """
    assert(radius > 0)

    # complex numbers make np.mgrid generate a list of interpolated values
    phi, theta = np.mgrid[0:2 * np.pi:complex(0, samples * 2), 0:np.pi:complex(0, samples)]
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    return x, y, z
