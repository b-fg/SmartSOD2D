import os
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import signal, interpolate

def force_latex(fontsize=20):
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font',family = 'sans-serif', size=fontsize)
    mpl.rc('xtick', labelsize=fontsize)
    mpl.rc('ytick', labelsize=fontsize)
    mpl.rcParams['axes.linewidth'] = 0.25


def plot_xy(x=None, y=None, fname=None, **kwargs):
    x_label = kwargs.pop('x_label', None)
    y_label = kwargs.pop('y_label', None)
    xylog = kwargs.pop('xylog', '')
    fmt = kwargs.pop('fmt', 'png')
    skip_every = kwargs.pop('skip_every', 1)
    return_data = kwargs.pop('return_data', False)
    return_figure = kwargs.pop('return_figure', False)
    xticks = kwargs.pop('xticks', None)
    yticks = kwargs.pop('yticks', None)
    fontsize = kwargs.pop('fontsize', 20)
    box = kwargs.pop('box', 1)
    force_latex(fontsize)
    if fontsize:
        plt.rc('font', family='sans-serif', size=fontsize)
        mpl.rc('xtick', labelsize=fontsize)
        mpl.rc('ytick', labelsize=fontsize)

    fig, ax = plt.subplots()

    if x is not None and y is not None:
        y_delta = np.max(np.abs([np.min(y), np.max(y)])) * 0.1
        x_range = kwargs.pop('x_range', (np.min(x), np.max(x)))
        y_range = kwargs.pop('y_range', (np.min(y) - y_delta, np.max(y) + y_delta))
        x_indices = np.where((x_range[0] <= x) & (x <= x_range[1]))
        ax.set_xlim(x_range[0], x_range[1])
        ax.set_ylim(y_range[0], y_range[1])
        x, y = x[x_indices][::skip_every], y[x_indices][::skip_every]
        if xylog == 'xy': ax.loglog(x, y, **kwargs)
        elif 'x' in xylog: ax.semilogx(x, y, **kwargs)
        elif 'y' in xylog: ax.semilogy(x, y, **kwargs)
        else: ax.plot(x, y, **kwargs)

    if xticks: ax.set_xticks(xticks)
    if yticks: ax.set_yticks(yticks)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.tick_params(bottom=True, top=True, right=True, which='both', direction='in', length=2)
    if box: ax.set_box_aspect(box)

    if return_figure:
        return fig, ax
    else:
        save_plot(fname)
    if return_data and (x is not None and y is not None):
        return x, y
    return

def save_plot(fname):
    plt.savefig(fname, format=fname.split('.')[-1], bbox_inches='tight') if fname else plt.show()
    plt.close()

def plot_tw(tw_file, fname=None, **kwargs):
    t, tw_neg, tw_pos = np.loadtxt(tw_file, delimiter=",", unpack=True)
    return plot_xy(x=t, y=tw_neg, fname=fname, **kwargs)


def plot_control_signal(control_file, fname=None, t_min=0.0, t_max=1e6):
    t, y = np.loadtxt(control_file, delimiter=",", unpack=True)
    return plot_xy(x=t, y=y, fname=fname, **kwargs)


def plot_witness_timeseries(witness_file="resultwit.h5", field='u_x', wit_idx=0, fname=None, **kwargs):
    h5file = h5py.File(witness_file)
    t = np.array(h5file.get('time'))
    field = np.array(h5file.get(field))[wit_idx]
    return plot_xy(x=t, y=field, fname=fname, **kwargs)


def welch(h5data, field='u_x', wit_idx=0, t_range=(0, np.inf), dt_resample=None, window='hann', n_splits=6):
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

def plot_witness_spectra_welch(fl, yl, fname_out=None, **kwargs):
    fig, ax = plot_xy(x=fl[0][1:], y=yl[0][1:], fname=fname_out, return_figure=True, xylog='xy', **kwargs)
    for f,y in zip(fl[1:], yl[1:]):
        ax.plot(f, y)
    xlog, ylog = loglogLine(p2=(0.2, 3e-4), p1x=3e-2, m=-5/3)
    ax.loglog(xlog, ylog, color='black', lw=1, ls='dotted')
    plt.annotate(r'$-5/3$', xy=(7e-2, 2e-3), fontsize=10)
    plt.tight_layout()
    save_plot(fname_out)

def witness_spectra_welch(h5data, field='u_x', wit_point=0, t_range=(0.0, np.inf), dt_resample=1.0,
        window='boxcar', n_splits=6): #, fname_out=None, **kwargs):
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
    return f, y
    # fig, ax = plot_xy(f[1:], y[1:], fname=fname_out, return_figure=True, xylog='xy', **kwargs)
    # xlog, ylog = loglogLine(p2=(0.2, 3e-4), p1x=3e-2, m=-5/3)
    # ax.loglog(xlog, ylog, color='black', lw=1, ls='dotted')
    # plt.annotate(r'$-5/3$', xy=(7e-2, 2e-3), fontsize=10)
    # xlog, ylog = loglogLine(p2=(0.02, 8e-3), p1x=2e-3, m=-3/5)
    # ax.loglog(xlog, ylog, color='black', lw=1, ls='dotted')
    # plt.annotate(r'$-3/5$', xy=(0.006,0.02), fontsize=10)
    # plt.tight_layout()
    # save_plot(fname_out)

def get_actions(control_drl_fname, control_smooth_fname, t_range=(0.0, np.inf)):
    data = np.loadtxt(control_smooth_fname, delimiter=",", unpack=False)
    t_smooth, a_smooth = data[:, 0], data[:, 1:][:,::2]
    t_indices = np.where((t_range[0] <= t_smooth) & (t_smooth <= t_range[1]))
    t_smooth, a_smooth = t_smooth[t_indices], a_smooth[t_indices]

    data = np.loadtxt(control_drl_fname, delimiter=",", unpack=False)
    t, a = data[:, 0], data[:, 1:][:,::2]
    t_indices = np.where((t_range[0] <= t) & (t <= t_range[1]))
    t, a = t[t_indices], a[t_indices]
    return t_smooth, a_smooth, t, a


def plot_actions(control_drl_fname, control_smooth_fname, t_range=(0.0, np.inf), t_scale=1, fname_out=None, kwargs_plots=None):
    t_smooth, a_smooth, t, a = get_actions(control_drl_fname, control_smooth_fname, t_range=t_range)
    actions_signal(t, a, t_smooth, a_smooth, t_scale=t_scale, fname_out=fname_out, **kwargs_plots['signals'])
    actions_autocorrelation(t, a, fname_out=fname_out, t_scale=t_scale, **kwargs_plots['R'])
    actions_powerspectrum(t, a, fname_out=fname_out, **kwargs_plots['PS'])


def actions_signal(t, a, t_smooth, a_smooth, t_scale=1, fname_out=None, **kwargs):
    def export_legend(legend, fname_out="actions_legend.pdf"):
        fig  = legend.figure
        fig.canvas.draw()
        bbox  = legend.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(fname_out, dpi="figure", bbox_inches=bbox)

    colors = kwargs.pop('colors', ['blue', 'orange', 'green'])
    legend = kwargs.pop('legend', True)
    marl_envs = a.shape[-1]

    fig, ax = plot_xy(x=t_smooth/t_scale, y=a_smooth[:, 0], return_figure=True, color=colors[0], **kwargs)
    ax.scatter(t/t_scale, a[:, 0], s=18, edgecolor='black', linewidth=0.3, color=colors[0], label=r'$v_\mathrm{ac,1}$', zorder=999)
    for i in range(1, marl_envs):
        ax.plot(t_smooth/t_scale, a_smooth[:, i], color=colors[i])
        ax.scatter(t/t_scale, a[:, i], s=18, edgecolor='black', linewidth=0.3, color=colors[i], label=r'$v_\mathrm{ac,'+str(i+1)+'}$', zorder=999)
    if legend:
        legend = ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.7),
            labelspacing=0.01, handleheight=0.5, handletextpad=0.0, fontsize=12)
        legend.get_frame().set_linewidth(0.0)
        for handle in legend.legend_handles:
            handle.set_sizes([20])
        export_legend(legend, fname_out="/".join(fname_out.split('/')[:-1]) + '/actions_legend.pdf')
        legend.remove()
        ax.set_yticklabels([])
        ax.set_yticks([])
        ax.set(ylabel=None)

    plt.axhline(y=0.0, color='black', linestyle='--', linewidth=0.5)
    save_plot(fname_out)
    print(f"Actions plot saved on: {fname_out}")

def actions_autocorrelation(t, a, t_scale=1, fname_out=None, **kwargs):
    def correlate(x, y):
        assert len(x) == len(y)
        xn = (x - np.mean(x))
        yn = (y - np.mean(y))
        c = np.correlate(xn, yn, mode='full') / (np.std(x) * np.std(y) * len(xn))
        return c[c.size//2:]

    fname, fmt = "".join(fname_out.split('.')[:-1]), fname_out.split('.')[-1]
    fname_out = fname + f'_AC.' + fmt
    colors = kwargs.pop('colors', ['blue', 'orange', 'green'])
    legend = kwargs.pop('legend', True)

    # Resample equidistant and subtract mean and
    dt_avg = np.mean(np.diff(t))
    print(f'Average dt = {dt_avg:.4f}')
    dt_resample = kwargs.pop('dt_resample', None)
    dt = dt_avg if dt_resample is None else dt_resample
    print(f'Resampling with dt = {dt:.4f} for t = [{t[0]:.4f}, {t[-1]:.4f}]')
    n_samples = int((t[-1] - t[0]) / dt)
    a_resampled = np.zeros((n_samples, a.shape[1]))
    t_equidistant = np.linspace(t[0], t[-1], n_samples)
    for i in range(a.shape[1]):
        y = a[:,i].flatten()
        f = interpolate.interp1d(t, y)
        a_resampled[:,i] = f(t_equidistant)
    a1, a2, a3 = np.hsplit(a_resampled, 3)
    a1 = a1.flatten(); a2 = a2.flatten(); a3 = a3.flatten()

    # Perform correlations
    c11 = correlate(a1, a1)
    c22 = correlate(a2, a2)
    c33 = correlate(a3, a3)
    c12 = correlate(a1, a2)
    c13 = correlate(a1, a3)
    c23 = correlate(a2, a3)
    tau = dt*np.array(range(n_samples))

    # Plot
    fig, ax = plot_xy(x=tau/t_scale, y=c11, label=r'$R_{11}$', fname=fname_out, return_figure=True, color=colors[0], **kwargs)
    for (corr, label, color) in [(c12, '12', colors[3]), (c22, '22', colors[1]), (c13, '13', colors[4]),
        (c33, '33', colors[2]), (c23, '23', colors[5])]:
        ax.plot(tau/t_scale, corr, label=r'$R_{'+label+r'}$', color=color)
    if legend:
        legend = ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.56), handlelength=2,
            labelspacing=0.05, handleheight=1, handletextpad=1, fontsize=kwargs['fontsize'])
        legend.get_frame().set_linewidth(0.0)
    plt.axhline(y=0.0, color='black', linestyle='--', linewidth=0.5)
    save_plot(fname_out)
    print(f"Actions plot saved on: {fname_out}")

def actions_powerspectrum(t, a, fname_out='actions.pdf', **kwargs):
    fname, fmt = "".join(fname_out.split('.')[:-1]), fname_out.split('.')[-1]
    window = kwargs.pop('window', 'boxcar')
    n_splits = kwargs.pop('n_splits', 6)
    fname_out = fname + f'_PS_{window}_{n_splits}.' + fmt
    colors = kwargs.pop('colors', ['blue', 'orange', 'green'])
    legend = kwargs.pop('legend', True)

    # Resample equidistant and subtract mean and
    dt_avg = np.mean(np.diff(t))
    print(f'Average dt = {dt_avg:.4f}')
    dt_resample = kwargs.pop('dt_resample', None)
    dt = dt_avg if dt_resample is None else dt_resample
    print(f'Resampling with dt = {dt:.4f} for t = [{t[0]:.4f}, {t[-1]:.4f}]')
    n_samples = int((t[-1] - t[0]) / dt)
    a_resampled = np.zeros((n_samples, a.shape[1]))
    t_equidistant = np.linspace(t[0], t[-1], n_samples)
    for i in range(a.shape[1]):
        y = a[:,i].flatten()
        f = interpolate.interp1d(t, y)
        a_resampled[:,i] = f(t_equidistant)
    a1, a2, a3 = np.hsplit(a_resampled, 3)

    freqs, Pxx_spec = signal.welch(a1.flatten(), 1/dt, window, a1.size//n_splits, scaling='spectrum')
    fig, ax = plot_xy(x=freqs[1:], y=Pxx_spec[1:], label=r'$v_{\mathrm{ac},1}$', fname=fname_out, return_figure=True,
        xylog='xy', color=colors[0], **kwargs)
    for i,a in enumerate([a2, a3]):
        freqs, Pxx_spec = signal.welch(a.flatten(), 1/dt, window, a.size//n_splits, scaling='spectrum')
        ax.plot(freqs, Pxx_spec, color=colors[i+1], label=r'$v_\mathrm{ac,'+str(i+2)+'}$')
    if legend:
        legend = ax.legend(loc='lower center', ncol=3, bbox_to_anchor=(0.4, -0.48), handlelength=1,
            labelspacing=0.1, handleheight=1, handletextpad=0.3, columnspacing=0.5, fontsize=kwargs['fontsize']+4)
        legend.get_frame().set_linewidth(0.0)

    xlog, ylog = loglogLine(p2=(3e-3, 2e-3), p1x=4e-4, m=-5/3)
    ax.loglog(xlog, ylog, color='black', lw=1, ls='dotted')
    plt.annotate(r'$-5/3$', xy=(1e-3,0.015), fontsize=kwargs['fontsize']-2)
    ax.xaxis.set_tick_params(pad=10)
    save_plot(fname_out)
    print(f"Actions plot saved on: {fname_out}")

def plot_lx(fname_in, t_scale=1, t_range=(0.0, np.inf), t_transient=5000, fname_out=None, **kwargs):
    data = np.loadtxt(fname_in, delimiter=',', unpack=False)
    t, lx = data[:, 0], np.mean(data[:, 1:], axis=1)
    t_indices = np.where((t_range[0] <= t) & (t <= t_range[1]))
    t, lx = t[t_indices], lx[t_indices]
    t, iu = np.unique(t, return_index=True)
    lx = lx[iu]
    fig, ax = plot_xy(x=t/t_scale, y=lx, fname=fname_out, return_figure=True,
        x_range=(t_range[0]/t_scale, t_range[1]/t_scale), color='black', **kwargs)

    t, lx = data[:, 0], np.mean(data[:, 1:], axis=1)
    t_indices = np.where((t_transient <= t) & (t <= t_range[1]))
    t, lx = t[t_indices], lx[t_indices]
    t, iu = np.unique(t, return_index=True)
    lx = lx[iu]

    print(f't_range lx avg = [{t[0]}, {t[-1]}]')
    lx_mean, _ = print_lx_stats(t, lx)
    ax.axhline(y=lx_mean, color='grey', linestyle='--')
    save_plot(fname_out)


def print_lx_stats(t, lx):
    lx_mean = np.average(lx, weights=t)
    lx_std = np.sqrt(1 / len(lx) * np.sum((lx - lx_mean)**2))
    print(f'mean(lx) = {lx_mean}\n std(lx) = {lx_std}')
    return lx_mean, lx_std


def plot_history(history, events_list, output_dir='.', reward_norm=145.0, n_actions=40, **kwargs):
    fig, ax = plot_xy(None, None, None, return_figure=True, **kwargs)
    history.reload()
    events = history.events
    color = iter(plt.cm.rainbow(np.linspace(0, 1, len(events_list))))
    for k,v in events_list.items():
        e = events[k]
        ax.plot(e['x'], e['y'] / n_actions, color=next(color), label=v)

    ax2 = ax.twinx()
    ax.set_xlim(*kwargs.pop('x_range', (np.min(e['x']), np.max(e['x']))))
    ax.set_ylim(-1.2, -0.7)
    ax.set_yticks([y for y in np.linspace(-0.7, -1.2,6)])
    ax.set_xlabel(r'$\mathrm{Episodes}$')
    ax.set_ylabel(r'$\mathrm{Reward}$')
    ax.set_box_aspect(1)

    ax2.set_ylabel(r'$\mathrm{Loss}$')
    ax2.set_ylim(0, 200)

    loss = events['Losses/total_abs_loss']
    ax2.plot(loss['x'], loss['y'], label=r'$\mathrm{Absolute\,\,Loss}$', color='black')

    legend = fig.legend(loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.2),
        labelspacing=0.5, handleheight=0.5, handletextpad=0.5, fontsize=16)
    legend.get_frame().set_linewidth(0.0)

    fname_out = os.path.join(output_dir, 'history.pdf')
    save_plot(fname_out)

    print(f"History plot saved on: {fname_out}")
