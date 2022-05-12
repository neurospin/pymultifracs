import time
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, Normalize
import seaborn as sns
import numpy as np
import pandas as pd

from .psd import log_plot


def plot_multiscale(results, seg2color, ax=None):

    ax = plt.gca() if ax is None else ax

    segs = {*results.index.get_level_values(1).unique()}
    subjects = results.index.get_level_values(0).unique()
    n = len(subjects)

    maxf = {
        seg: results.loc[pd.IndexSlice[:, seg], 'freq'].apply(max).min()
        for seg in segs
    }

    minf = {
        seg: results.loc[pd.IndexSlice[:, seg], 'freq'].apply(min).max()
        for seg in segs
    }

    trimmed = {
        seg: pd.Series([row[1].mscale[(row[1].freq <= maxf[seg])
                                      & (row[1].freq >= minf[seg])]
                        for row in results.loc[pd.IndexSlice[:, seg],
                                               ['freq', 'mscale']].iterrows()],
                       index=subjects)
        for seg in segs
    }

    stacks = {
        seg: np.vstack(trimmed[seg])
        for seg in segs
    }

    averages = {
        seg: stacks[seg].mean(axis=0)
        for seg in segs
    }

    freqs_avg = {
        seg: (freq := results.loc[subjects[0], seg].freq)[
            (freq <= maxf[seg]) & (freq >= minf[seg])]
        for seg in segs
    }

    freqs = {
        seg: results.loc[pd.IndexSlice[:, seg], 'freq']
        for seg in segs
    }

    mscales = {
        seg: results.loc[pd.IndexSlice[:, seg], 'mscale']
        for seg in segs
    }

    mscales = ([mscale for seg in segs
                for mscale in results.loc[pd.IndexSlice[:, seg], 'mscale']]
               + [*averages.values()])
    freqs = ([freq for seg in segs
              for freq in results.loc[pd.IndexSlice[:, seg], 'freq']]
             + [*freqs_avg.values()])

    color = ([seg2color[seg] for seg in segs for i in range(n)]
             + [seg2color[seg + '_avg'] for seg in segs])
    lw = ([1] * (len(segs) * n)) + ([2] * len(segs))

    log_plot(freqs, mscales, lowpass_freq=50, color=color, linewidth=lw, ax=ax)

    ticks = ax.get_xticks()
    labels = [f'{t:g}\n{2 ** t:.2g}' for t in ticks]

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)

    sns.despine()

    ax.set_xlabel('log2 f', fontsize=18, c='black')
    ylim = ax.get_ylim()
    ax.vlines(x=[results.iloc[0].slope[0][0], results.iloc[0].slope[0][-1]],
              ymin=ylim[0], ymax=ylim[1], linestyles='dashed',
              colors='#0000003f')
    ax.set_ylim(ylim)


def plot_cumulants(cm, fignum=1, nrow=3, j1=None, filename=None, scaling_range=0):
    """
    Plots the cumulants.
    Args:
    fignum(int):  figure number
    plt        :  pointer to matplotlib.pyplot
    """

    if j1 is None:
        j1 = cm.j.min()

    if cm.j.min() > j1:
        raise ValueError(f"Expected mrq to have minium scale {j1=}, got {cm.j.min()} instead")

    j_min = j1 - cm.j.min()

    nrow = min(nrow, len(cm.m))

    if len(cm.m) > 1:
        plot_dim_1 = nrow
        plot_dim_2 = int(np.ceil(len(cm.m) / nrow))

    else:
        plot_dim_1 = 1
        plot_dim_2 = 1

    fig, axes = plt.subplots(plot_dim_1,
                             plot_dim_2,
                             num=fignum,
                             squeeze=False)

    fig.suptitle(cm.formalism + r' - cumulants $C_m(j)$')

    x = cm.j[j_min:]

    for ind_m, m in enumerate(cm.m):

        y = getattr(cm, f'C{m}')[j_min:, scaling_range]

        if cm.bootstrapped_cm is not None:

            if cm.bootstrapped_cm.j.min() > j1:
                raise ValueError(f"Expected bootstrapped mrq to have minimum scale {j1=}, got {cm.bootstrapped_cm.j.min()} instead")

            CI = getattr(cm, f'CIE_C{m}')[j1 - cm.bootstrapped_cm.j.min():]

            CI -= y[:, None]
            CI[:, 1] *= -1
            assert (CI < 0).sum() == 0
            CI = CI.transpose()

        else:
            CI = None

        ax = axes[ind_m % nrow][ind_m // nrow]

        ax.errorbar(x, y, CI, fmt='r--.', zorder=-1)
        ax.set_xlabel('j')
        ax.set_ylabel('m = ' + str(m))
        # ax.grid()
        # plt.draw()

        if len(cm.log_cumulants) > 0:

            x0, x1 = cm.scaling_ranges[scaling_range]
            slope_log2_e = cm.log_cumulants[ind_m, scaling_range, 0]
            slope = cm.slope[ind_m, scaling_range, 0]
            intercept = cm.intercept[ind_m, scaling_range, 0]

            y0 = slope*x0 + intercept
            y1 = slope*x1 + intercept

            if cm.bootstrapped_cm is not None:
                CI = getattr(cm, f"CIE_c{m}")
                CI_legend = (f"; [{CI[scaling_range, 1]:.3f}, "
                             f"{CI[scaling_range, 0]:.3f}]")
            else:
                CI_legend = ""

            legend = (rf'$c_{m}$ = {slope_log2_e:.3f}' + CI_legend)

            ax.plot([x0, x1], [y0, y1], color='k',
                    linestyle='-', linewidth=2, label=legend, zorder=0)
            ax.legend()
            plt.draw()

    for j in range(ind_m + 1, len(axes.flat)):
        fig.delaxes(axes[j % nrow][j // nrow])

    if filename is not None:
        plt.savefig(filename)


def plot_coef(mrq, j1, j2, leader=True, ax=None, vmin=None, vmax=None, leader_idx_correction=True):

    min_all = min([np.nanmin(mrq[s]) for s in range(j1, j2+1) if s in mrq])
    
    if vmax is None:
        vmax = max([np.nanmax(mrq[s]) for s in range(j1, j2+1) if s in mrq])
    if vmin is None:
        vmin = min_all
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 7))

    for i, scale in enumerate(range(j1, j2 + 1)):
        
        if scale not in mrq:
            continue

        temp = mrq[scale][:, 0]
        
        X = (np.arange(temp.shape[0] + 1) + (1 if leader and leader_idx_correction else 0)) * (2 ** (scale - j1 + 1))
        X = np.tile(X[:, None], (1, 2))

        C = temp[:, None]
        
        Y = np.ones(X.shape[0]) * scale
        Y = np.stack([Y - .5, Y + .5]).transpose()
        
        if leader:
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = CenteredNorm(halfrange=max([abs(vmin), abs(vmax)]))
        
        cmap = mpl.cm.get_cmap('inferno').copy()
        cmap.set_bad('grey')
        
        qm = ax.pcolormesh(X, Y, C, cmap=cmap, norm=norm)
        
    ax.set_ylim(j1-.5, j2+.5)
    
    ax.set_xlabel('shift')
    ax.set_ylabel('scale')
    ax.set_facecolor('grey')
    plt.colorbar(qm, ax=ax)


# From pyvista, since https://github.com/pyvista/pyvista/issues/1125 is not yet
# fixed


"""Start xvfb from Python."""

XVFB_INSTALL_NOTES = """Please install Xvfb with:
Debian
$ sudo apt install libgl1-mesa-glx xvfb
CentOS / RHL
$ sudo yum install libgl1-mesa-glx xvfb
"""


def start_xvfb(wait=3, window_size=None):
    """Start the virtual framebuffer Xvfb.
    Parameters
    ----------
    wait : float, optional
        Time to wait for the virtual framebuffer to start.  Set to 0
        to disable wait.
    window_size : list, optional
        Window size of the virtual frame buffer.  Defaults to the
        default window size in ``rcParams``.
    """
    from pyvista import rcParams

    if os.name != 'posix':
        raise OSError('`start_xvfb` is only supported on Linux')

    if os.system('which Xvfb > /dev/null'):
        raise OSError(XVFB_INSTALL_NOTES)

    # use current default window size
    window_size_parm = '%dx%dx24' % tuple(rcParams['window_size'])
    display_num = ':99'
    os.system(f'Xvfb {display_num} -screen 0 {window_size_parm} '
              '> /dev/null 2>&1 &')
    os.environ['DISPLAY'] = display_num
    if wait:
        time.sleep(wait)
