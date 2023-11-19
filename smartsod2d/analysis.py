import os
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal, interpolate


def force_latex(fontsize=14):
    plt.rc('text', usetex=True )
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font',family = 'sans-serif', size=fontsize)
    mpl.rc('xtick', labelsize=fontsize)
    mpl.rc('ytick', labelsize=fontsize)
    mpl.rcParams['axes.linewidth'] = 0.25


def plot_xy(x, y, fname=None, **kwargs):
    y_delta = np.max(np.abs([np.min(y), np.max(y)])) * 0.1
    x_range = kwargs.pop('x_range', (np.min(x), np.max(x)))
    y_range = kwargs.pop('y_range', (np.min(y) - y_delta, np.max(y) + y_delta))
    x_label = kwargs.pop('x_label', None)
    y_label = kwargs.pop('y_label', None)
    xylog = kwargs.pop('xylog', '')
    fmt = kwargs.pop('fmt', 'png')
    skip_every = kwargs.pop('skip_every', 1)
    return_data = kwargs.pop('return_data', False)
    return_figure = kwargs.pop('return_figure', False)
    xticks = kwargs.pop('xticks', None)
    yticks = kwargs.pop('yticks', None)
    fontsize = kwargs.pop('fontsize', None)
    box = kwargs.pop('box', 1)

    if fontsize:
        plt.rc('font',family = 'sans-serif',  size=fontsize) # use 13(JFM) or 16(SNH)
        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)

    x_indices = np.where((x_range[0] <= x) & (x <= x_range[1]))
    x, y = x[x_indices][::skip_every], y[x_indices][::skip_every]
    fig, ax = plt.subplots()
    if xylog == 'xy': ax.loglog(x, y, **kwargs)
    elif 'x' in xylog: ax.semilogx(x, y, **kwargs)
    elif 'y' in xylog: ax.semilogy(x, y, **kwargs)
    else: ax.plot(x, y, **kwargs)

    if xticks: ax.set_xticks(xticks)
    if yticks: ax.set_yticks(yticks)

    ax.set_xlim(x_range[0], x_range[1])
    ax.set_ylim(y_range[0], y_range[1])
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    ax.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)
    ax.set_box_aspect(box)

    if not return_figure:
        plt.savefig(fname, format=fname.split('.')[-1], bbox_inches='tight') if fname else plt.show()
        plt.close()
    if return_data:
        return x, y
    elif return_figure:
        return fig, ax


def plot_tw(tw_file, fname=None, **kwargs):
    t, tw_neg, tw_pos = np.loadtxt(tw_file, delimiter=",", unpack=True)
    return plot_xy(t, tw_neg, fname=fname, **kwargs)


def plot_control_signal(control_file, fname=None, t_min=0.0, t_max=1e6):
    t, y = np.loadtxt(control_file, delimiter=",", unpack=True)
    return plot_xy(t, y, fname=fname, **kwargs)


def plot_witness_timeseries(witness_file="resultwit.h5", field='u_x', wit_idx=0, fname=None, **kwargs):
    h5file = h5py.File(witness_file)
    t = np.array(h5file.get('time'))
    field = np.array(h5file.get(field))[wit_idx]
    return plot_xy(t, field, fname=fname, **kwargs)


def welch(h5data, field='u_x', wit_idx=0, t_range=(0, np.inf), dt_resample=None, window='hann', n_splits=4):
    t_all = np.array(h5data.get('time'))
    t_indices = np.where((t_range[0] <= t_all) & (t_all <= t_range[1]))
    y_witness = np.array(h5data.get(field))
    y_all = y_witness[wit_idx]
    y, t = y_all[t_indices], t_all[t_indices]

    dt_avg = np.mean(np.diff(t))
    print(f'Average dt = {dt_avg:.4f}')
    dt = dt_avg if dt_resample is None else dt_resample
    print(f'Resampling with dt = {dt:.4f} for t = [{t[0]:.4f}, {t[-1]:.4f}]')
    n_samples = int((t[-1] - t[0]) / dt)

    f = interpolate.interp1d(t, y)
    t_equidistant = np.linspace(t[0], t[-1], n_samples)
    y_equidistant = f(t_equidistant)

    freqs, Pxx_spec = signal.welch(y_equidistant, 1/dt, window, y_equidistant.size//n_splits, scaling='spectrum')
    return freqs, np.sqrt(Pxx_spec)


def print_dominant_freqs(f, y, n=5):
    y_argsort = np.argsort(y)[::-1]
    f, y = f[y_argsort[:n]], y[y_argsort[:n]]
    print(f'Dominant frequencies:\n f = {f}\n y = {y}')


def print_witness(h5data):
    xyz = np.array(h5data.get('xyz'))
    for i, p in enumerate(xyz):
        print(i, p)


def loglogLine(p2, p1x, m):
    b = np.log10(p2[1])-m*np.log10(p2[0])
    p1y = p1x**m*10**b
    return [p1x, p2[0]], [p1y, p2[1]]


def find_wit_idx(h5data, xy):
    """Select an (x,y) point the corresponding witness indices"""
    xyz = np.array(h5data.get('xyz'))
    x_indices = np.where(np.abs(xy[0] - xyz[:,0]) < 0.01)[0]
    y_indices = np.where(np.abs(xy[1] - xyz[:,1]) < 0.01)[0]
    indices = x_indices[np.nonzero(np.in1d(x_indices, y_indices))[0]]
    if indices.size == 0: raise ValueError(f'Point {xy} not found.')
    print(f'Witness points near: {xy}. Found witness indices: {indices}')
    for i in indices:
        print(i, xyz[i])
    return indices


def plot_witness_spectra_welch(h5data, field='u_x', wit_point=0, t_range=(0.0, np.inf), dt_resample=1.0,
        window='hann', n_splits=3, fname_out=None, **kwargs):
    force_latex()

    if wit_point is not int:
        indices = find_wit_idx(h5data, wit_point)
        yl = []
        for i in indices:
            print(f'Computing PS of witness point {i}')
            f, yy = welch(h5data, field=field, wit_idx=i, t_range=t_range, dt_resample=dt_resample, window=window, n_splits=n_splits)
            yl.append(yy)
        y = np.mean(yl, axis=0)
        print_dominant_freqs(f, y)
    else:
        f, y = welch(h5data, field=field, wit_idx=wit_idx, t_range=t_range, dt_resample=dt_resample, window=window, n_splits=n_splits)
        print_dominant_freqs(f, y)

    fig, ax = plot_xy(f[1:], y[1:], fname=fname_out, return_figure=True, xylog='xy', **kwargs)
    xlog, ylog = loglogLine(p2=(0.1, 1e-4), p1x=1e-2, m=-5/3)
    ax.loglog(xlog, ylog, color='black', lw=1, ls='dotted')
    plt.annotate(r'$-5/3$', xy=(1.3e-2,6e-4), fontsize=10)
    plt.tight_layout()
    plt.savefig(fname_out, format=fname_out.split('.')[-1], bbox_inches='tight') if fname_out else plt.show()


def plot_actions(control_drl_fname, control_smooth_fname, fname_out=None, **kwargs):
    force_latex()

    data = np.loadtxt(control_smooth_fname, delimiter=",", unpack=False)
    t_smooth, a_smooth = data[:, 0], data[:, 1:][:,::2]
    data = np.loadtxt(control_drl_fname, delimiter=",", unpack=False)
    t, a = data[:, 0], data[:, 1:][:,::2]
    marl_envs = a.shape[-1]

    fig, ax = plot_xy(t_smooth, a_smooth[:, 0], return_figure=True, label=r'$a_1$', **kwargs)
    for i in range(1, marl_envs):
        a_str = f'a_{i}'
        ax.plot(t_smooth, a_smooth[:, i], label=r'$'+f'a_{i}'+'$', linewidth=0.7)
        ax.scatter(t, a[:, i], s=7, edgecolor='black', linewidth=0.3, zorder=999)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
    plt.axhline(y=0.0, color='black', linestyle='--', linewidth=0.5)
    plt.savefig(fname_out, format=fname_out.split('.')[-1], bbox_inches='tight') if fname_out else plt.show()


def plot_lx(fname_in, t_range=(0.0, np.inf), fname_out=None, **kwargs):
    force_latex()
    data = np.loadtxt(fname_in, delimiter=',', unpack=False)
    t, lx = data[:, 0], np.mean(data[:, 1:], axis=1)
    t_indices = np.where((t_range[0] <= t) & (t <= t_range[1]))
    t, lx = t[t_indices], lx[t_indices]
    t, iu = np.unique(t, return_index=True)
    lx = lx[iu]
    plot_xy(t, lx, fname=fname_out, **kwargs)
    print_lx_stats(t, lx)


def print_lx_stats(t, lx):
    lx_mean = np.average(lx, weights=t)
    lx_std = np.sqrt(1 / len(lx) * np.sum((lx - lx_mean)**2))
    print(f'mean(lx) = {lx_mean}\n std(lx) = {lx_std}')

