import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import matplotlib.colors
from Math import generate_sphere_samples
from matplotlib.colors import LinearSegmentedColormap

_BACKGROUND_COLOR = 'xkcd:grey'
_COLOR_MAP_NAME = 'blue_red'
_INVISIBLE_COLOR = (1.0, 1.0, 1.0, 0.0)

blue_red_color_map = {'red':   ((0.0, 0.0, 0.0), (0.5, 0.0, 0.1), (1.0, 1.0, 1.0)),
                      'green': ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                      'blue':  ((0.0, 0.0, 1.0), (0.5, 0.1, 0.0), (1.0, 0.0, 0.0))}

plt.register_cmap(cmap=LinearSegmentedColormap('blue_red', blue_red_color_map))


def _fig_create_color_bar(fig, color_values, color_map_name=_COLOR_MAP_NAME):
    color_map = matplotlib.cm.get_cmap(color_map_name)
    min_value = color_values.min()
    max_value = color_values.max()
    color_bar = cm.ScalarMappable(cmap=color_map, norm=matplotlib.colors.Normalize(vmin=min_value, vmax=max_value))
    color_bar.set_array(np.ndarray([]))
    fig.colorbar(color_bar, shrink=0.5, aspect=10)


def _ax_create_3d_surface_cartesian_color(ax, x, y, z, w):
    face_colors = w / 255.0
    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=face_colors, linewidth=0, antialiased=False, shade=False)


def _ax_create_3d_surface_cartesian(ax, x, y, z, w=None, color_map_name=None):
    face_colors = None
    if color_map_name is not None and w is not None:
        color_map = matplotlib.cm.get_cmap(color_map_name)
        min_value = w.min()
        max_value = w.max()
        normalizer = cm.ScalarMappable(cmap=color_map, norm=matplotlib.colors.Normalize(vmin=min_value, vmax=max_value))
        face_colors = normalizer.to_rgba(w)

    ax.plot_surface(x, y, z, rstride=1, cstride=1, facecolors=face_colors, linewidth=0, antialiased=False, shade=False)


def create_3d_plot_with_color_map(x, y, z, w, title: str,  dest_filename=None, display=False, hide_text=False):
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.gca(projection='3d')

    fig.patch.set_facecolor(_BACKGROUND_COLOR)
    ax.set_facecolor(_BACKGROUND_COLOR)

    _ax_create_3d_surface_cartesian(ax, x, y, z, w, _COLOR_MAP_NAME)

    if hide_text:
        ax.xaxis.set_pane_color(_INVISIBLE_COLOR)
        ax.yaxis.set_pane_color(_INVISIBLE_COLOR)
        ax.zaxis.set_pane_color(_INVISIBLE_COLOR)
        ax.w_xaxis.line.set_color(_INVISIBLE_COLOR)
        ax.w_yaxis.line.set_color(_INVISIBLE_COLOR)
        ax.w_zaxis.line.set_color(_INVISIBLE_COLOR)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    else:
        plt.title(title)

        ax.set_xlabel('X axis', fontsize='large', labelpad=10)
        ax.set_ylabel('Y axis', fontsize='large', labelpad=10)
        ax.set_zlabel('Z axis', fontsize='large', labelpad=10)
        _fig_create_color_bar(fig, w, _COLOR_MAP_NAME)

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_box_aspect((1, 1, 1))

    plt.tight_layout()
    if dest_filename is not None:
        plt.savefig(dest_filename)

    if display:
        plt.show()

    plt.close(fig)


def create_3d_plot_with_color(x, y, z, w, title: str,  dest_filename=None, display=False, hide_text=False):
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.gca(projection='3d')

    fig.patch.set_facecolor(_BACKGROUND_COLOR)
    ax.set_facecolor(_BACKGROUND_COLOR)

    _ax_create_3d_surface_cartesian_color(ax, x, y, z, w)

    if hide_text:
        ax.xaxis.set_pane_color(_INVISIBLE_COLOR)
        ax.yaxis.set_pane_color(_INVISIBLE_COLOR)
        ax.zaxis.set_pane_color(_INVISIBLE_COLOR)
        ax.w_xaxis.line.set_color(_INVISIBLE_COLOR)
        ax.w_yaxis.line.set_color(_INVISIBLE_COLOR)
        ax.w_zaxis.line.set_color(_INVISIBLE_COLOR)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    else:
        plt.title(title)

        ax.set_xlabel('X axis', fontsize='large', labelpad=10)
        ax.set_ylabel('Y axis', fontsize='large', labelpad=10)
        ax.set_zlabel('Z axis', fontsize='large', labelpad=10)

    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)
    ax.set_box_aspect((1, 1, 1))

    plt.tight_layout()
    if dest_filename is not None:
        plt.savefig(dest_filename)

    if display:
        plt.show()

    plt.close(fig)


def test():
    x, y, z = generate_sphere_samples(samples=25)
    g_x, g_y = np.gradient(z)
    g = (g_x ** 2 + g_y ** 2) ** .5
    w = g/g.max()

    create_3d_plot_with_color_map(x, y, z, w, 'test figure, gradient of z',
                                  dest_filename='..//figures//test.png', display=True, hide_text=False)


if __name__ == '__main__':
    test()
