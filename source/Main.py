from Visualization import create_3d_plot_with_color_map
from Math import generate_sphere_samples, generate_sphere_samples_spherical_coordinates
from SphericalHarmonics import get_sh_basis_value_cartesian, get_sh_basis_value_radius_in_cartesian


def generate_sphere_visualization():
    x, y, z = generate_sphere_samples(1.0, 50)
    for order in [0, 1, 2, 3]:
        for degree in range(-order, order + 1):
            basis_value = get_sh_basis_value_cartesian(x, y, z, order, degree, normalize=False)
            figure_name = f'SH basis order {order} degree {degree}'
            file_name = f'..//figures//sh_{order}_{degree}'
            create_3d_plot_with_color_map(x, y, z, basis_value, figure_name,
                                          dest_filename=file_name, display=True, hide_text=False)


def generate_lobe_visualization():
    phi, theta = generate_sphere_samples_spherical_coordinates(50)
    for order in [0, 1, 2, 3]:
        for degree in range(-order, order + 1):
            x, y, z, w = get_sh_basis_value_radius_in_cartesian(phi, theta, order, degree)
            figure_name = f'SH basis order {order} degree {degree}'
            file_name = f'..//figures//sh_lobe_{order}_{degree}'
            create_3d_plot_with_color_map(x, y, z, w, figure_name,
                                          dest_filename=file_name, display=True, hide_text=False)


if __name__ == '__main__':
    generate_lobe_visualization()
    generate_sphere_visualization()
