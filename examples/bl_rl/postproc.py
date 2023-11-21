import os
import numpy as np
import h5py
from smartsod2d import analysis
from smartsod2d.utils import read_witness_points_coordinates

def witness_spectra_plot(fname_in, fname_out=None, wit_xyz_fname="witness.txt", wit_points=None, field='u_x', n_splits=6,
        t_range=(0, np.inf), fmt='.png', **kwargs):
    fname_in = fname_in if os.path.isabs(fname_in) else os.path.join(work_dir, fname_in)
    fname_out = fname_out if fname_out else field + fmt
    fname_out = fname_out if os.path.isabs(fname_out) else os.path.join(work_dir, fname_out)
    wit_xyz_fname = wit_xyz_fname if os.path.isabs(wit_xyz_fname) else os.path.join(work_dir, wit_xyz_fname)

    if wit_points is None:
        xyz = read_witness_points_coordinates(os.path.join(work_dir, wit_xyz_fname))
        wit_points = [(p[0], p[1]) for p in xyz[:n_xy_witness_points_planes]]
    elif isinstance(wit_points, tuple):
        wit_points = [wit_points]

    for wit_point in wit_points:
        wit_str = str(wit_point) if wit_point is int else f'x{wit_point[0]:.0f}y{wit_point[1]:.0f}'
        fout = field if not fname_out else fname_out
        fname_out = os.path.join(work_dir, fout + '_' + wit_str + fmt)
        h5data = h5py.File(os.path.join(work_dir, fname_in))
        analysis.plot_witness_spectra_welch(h5data, field=field, t_range=t_range, wit_point=wit_point, n_splits=n_splits,
                fname_out=fname_out, **kwargs)
        print(f"Plot saved on: {fname_out}")


def lx_plot(fname_in, fname_out="lx.pdf", t_range=(0, np.inf), **kwargs):
    fname_in = fname_in if os.path.isabs(fname_in) else os.path.join(work_dir, fname_in)
    fname_out = fname_out if os.path.isabs(fname_out) else os.path.join(work_dir, fname_out)

    analysis.plot_lx(fname_in, t_range=t_range, fname_out=fname_out,
        x_label=r'$t$', y_label=r'$l_x^*$', **kwargs)
    print(f"Plot saved on: {fname_out}")


def actions_plot(control_drl_fname, control_smooth_fname, fname_out="actions.pdf", t_range=(0, np.inf), **kwargs):
    control_drl_fname = control_drl_fname if os.path.isabs(control_drl_fname) else os.path.join(work_dir, control_drl_fname)
    control_smooth_fname = control_smooth_fname if os.path.isabs(control_smooth_fname) else os.path.join(work_dir, control_smooth_fname)
    fname_out = fname_out if os.path.isabs(fname_out) else os.path.join(work_dir, fname_out)
    analysis.plot_actions(control_drl_fname, control_smooth_fname, t_range=t_range, fname_out=fname_out, **kwargs)
    print(f"Plot saved on: {fname_out}")


# Main
work_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "output_0")
fmt = '.png'
t_range = (5000, 20000)
field = 'u_x'
n_splits = 6
witness_h5file = "resultwit.h5"
wit_xyz_fname = "witness.txt"
n_xy_witness_points_planes = 36
witness_point = (324, 15)
lx_fname = os.path.join(work_dir, "lx.txt")
control_drl_fname =  os.path.join(work_dir, "control_action.txt")
control_smooth_fname =  os.path.join(work_dir, "control_action_smooth.txt")

# witness_spectra_plot(witness_h5file, wit_xyz_fname=wit_xyz_fname, wit_points=witness_point, n_splits=n_splits,
    # fmt='.pdf', x_label=r'$f$', y_label=r'$\mathrm{PS}(u)$', color='black')

lx_plot(lx_fname, fname_out="lx.pdf", t_range=t_range, y_range=(40,280), color='black')

actions_plot(control_drl_fname, control_smooth_fname, fname_out="actions.pdf", t_range=t_range,
    y_range=(-0.4, 0.4), fontsize=11, linewidth=0.7, box=0.25)