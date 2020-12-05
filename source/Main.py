from Visualization import create_3d_plot_with_color_map
from Math import generate_sphere_samples
from SphericalHarmonics import get_sh_basis_value_cartesian


def main():
    x, y, z = generate_sphere_samples(1.0, 100)
    for order in [0, 1, 2]:
        for degree in range(-order, order + 1):
            basis_value = get_sh_basis_value_cartesian(x, y, z, order, degree, normalize=False)
            figure_name = f'SH basis order {order} degree {degree}'
            file_name = f'..//figures//sh_{order}_{degree}'
            create_3d_plot_with_color_map(x, y, z, basis_value, figure_name, dest_filename=file_name, display=False)


if __name__ == '__main__':
    main()
