import numpy as np


def generate_sphere_samples_spherical_coordinates(samples=25):
    phi, theta = np.mgrid[0:2 * np.pi:complex(0, samples * 2), 0:np.pi:complex(0, samples)]
    return phi, theta


def spherical_to_cartesian(phi, theta, radius):
    phi, theta = np.asarray(phi), np.asarray(theta)
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return x, y, z


def generate_sphere_samples(radius=1.0, samples=25):
    """
    Generate sphere samples using spherical coordinates with a grid of angles, the convention used is:
    r in (0, infinity)
    theta in (0, pi), rotating from the polar axis (z) (polar angle)
    phi in (0, 2pi), rotates around the z axis
    :param radius: radius of sphere
    :param samples: number of samples to use for each angle, creates a grid of sample * (2 * samples)
    :return: x,y,z arrays
    """
    assert(radius > 0 and samples > 0)

    # complex numbers make np.mgrid generate a list of interpolated values
    phi, theta = generate_sphere_samples_spherical_coordinates(samples)
    x, y, z = spherical_to_cartesian(phi, theta, radius)
    return x, y, z


def generate_angles_from_flattened_envmap(image):
    height = image.shape[0]
    width = image.shape[1]
    phi, theta = np.mgrid[0:2 * np.pi:complex(0, width) - np.pi, 0:np.pi:complex(0, height)]
    return phi, theta


def get_lambert_diffuse_from_envmap(image):
    height = image.shape[0]
    width = image.shape[1]
    phi, theta = np.mgrid[0:2 * np.pi:complex(0, width) - np.pi, 0:np.pi:complex(0, height)]
    x, y, z = spherical_to_cartesian(phi, theta, 1.0)
    input_light = np.transpose(image, [1, 0, 2]) / 255.0
    result = np.zeros((width, height, 3))

    sin_theta = np.sin(theta)
    sin_theta = sin_theta[:, :, np.newaxis]
    sin_theta = np.concatenate((sin_theta, sin_theta, sin_theta), axis=2)

    light_vector = np.concatenate((x[:, :, np.newaxis], y[:, :, np.newaxis], z[:, :, np.newaxis]), axis=2)

    # Riemann sum but divide by 2 to account for max(cos, 0) values that get ignored
    delta_area = np.pi * np.pi / (height * width)

    for w in range(0, width):
        for h in range(0, height):
            normal_vector = np.concatenate((
                np.tile(x[w, h], (width, height, 1)),
                np.tile(y[w, h], (width, height, 1)),
                np.tile(z[w, h], (width, height, 1))), axis=2)

            cos_angle = np.fmax(np.einsum('ijk,ijk->ij', light_vector, normal_vector), 0.0)

            cos_angle = cos_angle[:, :, np.newaxis]
            cos_angle = np.concatenate((cos_angle, cos_angle, cos_angle), axis=2)

            # Lambert diffuse in the rendering equation
            color = np.sum(input_light * cos_angle * sin_theta, axis=(0, 1)) * delta_area
            result[w, h, :] = np.fmin(color, 1.0)

    return np.transpose(result, [1, 0, 2])
