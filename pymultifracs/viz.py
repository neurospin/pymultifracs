import time
import os

import matplotlib.pyplot as plt
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


def plot_cumulants(cm, fignum=1, nrow=3, filename=None, cm_boot=None,
                   scaling_range=0):
    """
    Plots the cumulants.
    Args:
    fignum(int):  figure number
    plt        :  pointer to matplotlib.pyplot
    """

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

    x = cm.j

    for ind_m, m in enumerate(cm.m):

        y = getattr(cm, f'C{m}')

        if cm_boot is not None:
            CI = getattr(cm_boot, f'CIE_C{m}')(cm)

            CI -= y
            CI[:, 1] *= -1
            assert (CI < 0).sum() == 0
            CI = CI.transpose()

        else:
            CI = None

        ax = axes[ind_m % nrow][ind_m // nrow]

        # import ipdb; ipdb.set_trace()
        ax.errorbar(x[2:], y[2:, 0], CI[:, 2:], fmt='r--.', zorder=-1)
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

            if cm_boot is not None:
                CI = getattr(cm_boot, f"CIE_c{m}")(cm)
                # import ipdb; ipdb.set_trace()
                CI_legend = (f"; [{CI[scaling_range, 0]:.3f}, "
                             f"{CI[scaling_range, 1]:.3f}]")
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
