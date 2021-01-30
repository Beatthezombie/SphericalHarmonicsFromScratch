from Visualization import *
from Math import *
from SphericalHarmonics import *
from PIL import Image
import numpy as np

_SAMPLE_COUNT = 100
_DEGREE = [0, 1, 2, 3]


def generate_sphere_visualization():
    x, y, z = generate_sphere_samples(1.0, _SAMPLE_COUNT)
    for degree in _DEGREE:
        for order in range(-degree, degree + 1):
            basis_value = get_sh_basis_value_cartesian(x, y, z, degree, order)
            figure_name = f'SH basis degree {degree} order {order}'
            file_name = f'..//figures//sh_{degree}_{order}'
            create_3d_plot_with_color_map(x, y, z, basis_value, figure_name,
                                          dest_filename=file_name, display=False, hide_text=False)


def generate_lobe_visualization():
    phi, theta = generate_sphere_samples_spherical_coordinates(_SAMPLE_COUNT)
    for degree in _DEGREE:
        for order in range(-degree, degree + 1):
            x, y, z, w = get_sh_basis_value_radius_in_cartesian(phi, theta, degree, order)
            figure_name = f'SH basis degree {degree} order {order}'
            file_name = f'..//figures//sh_lobe_{degree}_{order}'
            create_3d_plot_with_color_map(x, y, z, w, figure_name,
                                          dest_filename=file_name, display=False, hide_text=True)


def sh_envmap_approximation(image_path, ):
    image = Image.open(image_path)
    image = np.asarray(image)
    generate_angles_from_flattened_envmap(image)


def envmap_visualization(image_path):
    image = Image.open(image_path)
    image = np.asarray(image)
    # phi, theta = generate_angles_from_flattened_envmap(image)
    # x, y, z = spherical_to_cartesian(phi, theta, 1.0)

    colors = get_lambert_diffuse_from_envmap(image)
    colors = colors * 255
    colors = colors.astype(np.uint8)
    image = Image.fromarray(colors)
    image.save(f'..//envmaps//factory//lambert_diffuse.jpg', "JPEG")

    # colors = np.transpose(image, [1, 0, 2])
    # figure_name = 'envmap on sphere'
    # file_name = f'..//figures//color_test'
    # create_3d_plot_with_color(x, y, z, colors, figure_name, dest_filename=file_name, display=True, hide_text=False)


def sh_approximation(image_path):
    image = Image.open(image_path)
    image = np.asarray(image)

    height = image.shape[0]
    width = image.shape[1]
    coefficient_image = np.zeros((width, height, 3))
    degrees = [0, 1, 2, 3]

    for degree in degrees:
        for order in range(-degree, degree + 1):
            coefficient_image = coefficient_image + compute_sh_coefficient_from_image(image, degree, order)
    coefficient_image = np.clip(coefficient_image, 0.0, 1.0)

    result = np.transpose(coefficient_image, [1, 0, 2]) * 255.0
    result = result.astype(np.uint8)
    result = Image.fromarray(result)

    result.save(f'..//envmaps//road//lambert_diffuse_sh_{max(degrees)}.jpg', "JPEG")


if __name__ == '__main__':

    # image source: http://www.hdrlabs.com/sibl/archive.html

    # generate_lobe_visualization()
    generate_sphere_visualization()
    # envmap_visualization('..//envmaps//factory//factory_tiny.jpg')
    # sh_approximation('..//envmaps//road//lambert_diffuse.jpg')
