from distutils.log import error
import time
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm, Normalize, PowerNorm
import matplotlib.cm as cm
from matplotlib.ticker import AutoLocator
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


def cp_string_format(cp, CI=False):

    threshold_1 = .2 if CI else .1
    threshold_2 = .01 if CI else .001

    if abs(cp) > threshold_1:
        return f"{cp:.2g}"
    elif abs(cp) > threshold_2:
        return f"{cp:.1g}"
    elif CI:
        return f"{cp:.2f}"
    else:
        return f"{cp:.3f}"


def plot_bicm(cm, ind_m1, ind_m2, j1, j2, scaling_range, ax, C_color='grey',
              fit_color='k', plot_legend=False, lw_fit=2, plot_fit=True,
              C_fmt='--.', lw_C=None, offset=0, plot_CI=True, **C_kwargs):

    if j1 is None:
        j1 = cm.j.min()

    if j2 is None:
        j2 = cm.j.max()

    if cm.j.min() > j1:
        raise ValueError(f"Expected mrq to have minium scale {j1=}, got "
                         f"{cm.j.min()} instead")

    j_min = j1 - cm.j.min()
    j_max = j2 - cm.j.min() + 1

    m1 = cm.m[ind_m1]
    m2 = cm.m[ind_m2]

    x = cm.j[j_min:j_max]
    y = getattr(cm, f'C{m1}{m2}')[j_min:j_max]

    if cm.bootstrapped_mrq is not None and plot_CI:

        if cm.bootstrapped_mrq.j.min() > j1:
            raise ValueError(
                f"Expected bootstrapped mrq to have minimum scale {j1=}, got "
                f"{cm.bootstrapped_mrq.j.min()} instead")

        CI = getattr(cm, f'CIE_C{m1}{m2}')[
            j1 - cm.bootstrapped_mrq.j.min():
            j2 - cm.bootstrapped_mrq.j.min() + 1]

        CI -= y[:, None]
        CI[:, :, 1] *= -1
        assert (CI < 0).sum() == 0
        CI = CI.transpose()

    else:
        CI = None

    y += offset

    errobar_params = {
        'zorder': -1
    }

    errobar_params.update(C_kwargs)

    ax.errorbar(x, y, CI, fmt=C_fmt, color=C_color, lw=lw_C, **errobar_params)

    ax.set_xlabel('j')
    ax.set_ylabel(f'$C_{m1}{m2}(j)$')
    # ax.grid()
    # plt.draw()

    if len(cm.log_cumulants) > 0 and plot_fit:

        x0, x1 = cm.scaling_ranges[scaling_range]
        slope_log2_e = cm.log_cumulants[ind_m1, ind_m2, scaling_range]
        slope = cm.slope[ind_m1, ind_m2, scaling_range]
        intercept = cm.intercept[ind_m1, ind_m2, scaling_range]

        y0 = slope*x0 + intercept
        y1 = slope*x1 + intercept

        if cm.bootstrapped_mrq is not None:
            CI = getattr(cm, f"CIE_c{m1}{m2}")
            CI_legend = (
                f"; [{cp_string_format(CI[scaling_range, 1], True)}, "
                f"{cp_string_format(CI[scaling_range, 0], True)}]")
        else:
            CI_legend = ""

        legend = (rf'$c_{{{m1}{m2}}}$ = {cp_string_format(slope_log2_e)}'
                  + CI_legend)

        ax.plot([x0, x1], [y0, y1], color=fit_color,
                linestyle='-', linewidth=lw_fit, label=legend, zorder=0)
        if plot_legend:
            ax.legend()


def plot_cm(cm, ind_m, j1, j2, scaling_range, ax, C_color='grey',
            fit_color='k', plot_legend=False, lw_fit=2, plot_fit=True,
            C_fmt='--.', lw_C=None, offset=0, plot_CI=True, signal_idx=0,
            **C_kwargs):

    if j1 is None:
        if cm.bootstrapped_mrq is not None:
            j1 = cm.bootstrapped_mrq.j.min()
        else:
            j1 = cm.j.min()


    if j2 is None:
        j2 = cm.j.max()

    if cm.j.min() > j1:
        raise ValueError(f"Expected mrq to have minium scale {j1=}, got "
                         f"{cm.j.min()} instead")

    j_min = int(j1 - cm.j.min())
    j_max = int(j2 - cm.j.min() + 1)

    m = cm.m[ind_m]

    x = cm.j[j_min:j_max]

    y = getattr(cm, f'C{m}')[j_min:j_max, signal_idx, 0]

    if cm.bootstrapped_mrq is not None and plot_CI:

        if cm.bootstrapped_mrq.j.min() > j1:
            raise ValueError(
                f"Expected bootstrapped mrq to have minimum scale {j1=}, got "
                f"{cm.bootstrapped_mrq.j.min()} instead")

        CI_slice = np.s_[int(j1 - cm.bootstrapped_mrq.j.min()):
                         int(j2 - cm.bootstrapped_mrq.j.min() + 1)]

        CI = getattr(cm, f'CIE_C{m}')[CI_slice, signal_idx]

        CI -= y[:, None]
        CI[:, 1] *= -1
        assert (CI < 0).sum() == 0
        CI = CI.transpose()

    else:
        CI = None

    y += offset

    errobar_params = {
        'zorder': -1
    }

    errobar_params.update(C_kwargs)

    ax.errorbar(x, y, CI, fmt=C_fmt, color=C_color, lw=lw_C, **errobar_params)

    ax.set_xlabel('j')
    ax.set_ylabel(f'$C_{m}(j)$')
    # ax.grid()
    # plt.draw()

    if len(cm.log_cumulants) > 0 and plot_fit:

        x0, x1 = cm.scaling_ranges[scaling_range]
        slope_log2_e = cm.log_cumulants[ind_m, scaling_range, signal_idx]
        slope = cm.slope[ind_m, scaling_range, signal_idx]
        intercept = cm.intercept[ind_m, scaling_range, signal_idx]

        y0 = slope*x0 + intercept
        y1 = slope*x1 + intercept

        if cm.bootstrapped_mrq is not None:
            CI = getattr(cm, f"CIE_c{m}")[scaling_range, signal_idx]

            CI_legend = (
                f"; [{cp_string_format(CI[1], True)}, "
                f"{cp_string_format(CI[0], True)}]")
        else:
            CI_legend = ""

        legend = (rf'$c_{m}$ = {cp_string_format(slope_log2_e)}'
                  + CI_legend)

        ax.plot([x0, x1], [y0, y1], color=fit_color,
                linestyle='-', linewidth=lw_fit, label=legend, zorder=0)
        if plot_legend:
            ax.legend()


def plot_cumulants(cm, figsize, fignum=1, nrow=3, j1=None, filename=None,
                   scaling_range=0, legend=True, n_cumul=None, signal_idx=0,
                   **kw):
    """
    Plots the cumulants.
    Args:
    fignum(int):  figure number
    plt        :  pointer to matplotlib.pyplot
    """

    if n_cumul is None:
        n_cumul = len(cm.m)

    nrow = min(nrow, n_cumul)

    if len(cm.m) > 1:
        plot_dim_1 = nrow
        plot_dim_2 = int(np.ceil(len(cm.m) / nrow))

    else:
        plot_dim_1 = 1
        plot_dim_2 = 1

    fig, axes = plt.subplots(plot_dim_1,
                             plot_dim_2,
                             num=fignum,
                             squeeze=False,
                             figsize=figsize,
                             sharex=True)

    # fig.suptitle(cm.formalism + r' - cumulants $C_m(j)$')

    # x = cm.j[j_min:]

    for ind_m, m in enumerate(cm.m[:n_cumul]):

        ax = axes[ind_m % nrow][ind_m // nrow]

        plot_cm(cm, ind_m, j1, None, scaling_range, ax, plot_legend=True,
                signal_idx=signal_idx, **kw)

        # y = getattr(cm, f'C{m}')[j_min:, scaling_range]

        # if cm.bootstrapped_mrq is not None:

        #     if cm.bootstrapped_mrq.j.min() > j1:
        #         raise ValueError(f"Expected bootstrapped mrq to have minimum scale {j1=}, got {cm.bootstrapped_mrq.j.min()} instead")

        #     CI = getattr(cm, f'CIE_C{m}')[j1 - cm.bootstrapped_mrq.j.min():]

        #     CI -= y[:, None]
        #     CI[:, 1] *= -1
        #     assert (CI < 0).sum() == 0
        #     CI = CI.transpose()

        # else:
        #     CI = None

        # ax = axes[ind_m % nrow][ind_m // nrow]

        # ax.errorbar(x, y, CI, fmt='--.', color='grey', zorder=-1)
        # ax.set_xlabel('j')
        # ax.set_ylabel('m = ' + str(m))
        # # ax.grid()
        # # plt.draw()

        # if len(cm.log_cumulants) > 0:

        #     x0, x1 = cm.scaling_ranges[scaling_range]
        #     slope_log2_e = cm.log_cumulants[ind_m, scaling_range, 0]
        #     slope = cm.slope[ind_m, scaling_range, 0]
        #     intercept = cm.intercept[ind_m, scaling_range, 0]

        #     y0 = slope*x0 + intercept
        #     y1 = slope*x1 + intercept

        #     if cm.bootstrapped_mrq is not None:
        #         CI = getattr(cm, f"CIE_c{m}")
        #         CI_legend = (
        #             f"; [{cp_string_format(CI[scaling_range, 1], True)}, "
        #             f"{cp_string_format(CI[scaling_range, 1], True)}]")
        #     else:
        #         CI_legend = ""

        #     legend = (rf'$c_{m}$ = {cp_string_format(slope_log2_e)}'
        #               + CI_legend)

        #     ax.plot([x0, x1], [y0, y1], color='k',
        #             linestyle='-', linewidth=2, label=legend, zorder=0)
        #     if legend:
        #         ax.legend()
        #     plt.draw()

    for j in range(ind_m):
        axes[j % nrow][j // nrow].xaxis.set_visible(False)

    for j in range(ind_m + 1, len(axes.flat)):
        fig.delaxes(axes[j % nrow][j // nrow])

    plt.tight_layout()

    if filename is not None:
        plt.savefig(filename)

    # return fig


def plot_coef(mrq, j1, j2, leader=True, ax=None, vmin=None, vmax=None,
              leader_idx_correction=True, cbar=True, figsize=(20, 7),
              gamma=.3, nan_idx=None):

    min_all = min([np.nanmin(np.abs(mrq[s])) for s in range(j1, j2+1) if s in mrq])

    if vmax is None:
        vmax = max([np.nanmax(mrq[s]) for s in range(j1, j2+1) if s in mrq])
    if vmin is None:
        vmin = min_all

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')#, width_ratios=[20, 1])

    norm = PowerNorm(vmin=vmin, vmax=vmax, gamma=gamma)

    # ax = axes[0]

    cmap = mpl.cm.get_cmap('inferno').copy()
    cmap.set_bad('grey')

    for i, scale in enumerate(range(j1, j2 + 1)):

        if scale not in mrq:
            continue

        temp = mrq[scale][:, 0]

        X = ((np.arange(temp.shape[0] + 1)
              + (1 if leader and leader_idx_correction else 0))
             * (2 ** (scale - j1 + 1)))
        X = np.tile(X[:, None], (1, 2))

        C = temp[:, None]

        if not leader:
            C = np.abs(C)

        Y = np.ones(X.shape[0]) * scale
        Y = np.stack([Y - .5, Y + .5]).transpose()

        qm = ax.pcolormesh(X, Y, C, cmap=cmap, norm=norm, rasterized=True)

        if nan_idx is not None:
            idx = np.unique(np.r_[nan_idx[scale], nan_idx[scale] + 1])

            segments = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

            for seg in segments:

                if len(seg) == 0:
                    continue

                ax.pcolormesh(X[seg[[0, -1]]], Y[seg[[0, -1]]], C[[0]], alpha=1,
                              edgecolor='xkcd:blue')

    ax.set(ylim=(j1-.5, j2+.5), yticks=range(j1, j2+1),
           xlabel='shift $k$', ylabel='scale $j$', facecolor='grey', xlim=(0, mrq[j1].shape[0]*2))

    ax.tick_params(which='minor', left=True, right=False, top=False, color='w')
    ax.yaxis.set_minor_locator(mpl.ticker.IndexLocator(base=1, offset=.5))
    ax.tick_params(which='major', right=False, top=False, color='w')

    # ax.set_yticks(range(j1, j2+1))
    # ax.set_xlabel('shift')
    # ax.set_ylabel('scale')
    # ax.set_facecolor('grey')

    if cbar:
        # formatter = mpl.ticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
        # formatter = mpl.ticker.LogFormatterSciNotation(labelOnlyBase=False, minor_thresholds=(np.inf, np.inf))
        locator = mpl.ticker.MaxNLocator(4, symmetric=False)
        cb = plt.colorbar(qm, ax=ax, ticks=locator, fraction=.1, aspect=8)
        # plt.colorbar(qm, ax=ax    es[0], ticks=locator, aspect=1)
        cb.ax.tick_params(which='major', size=3)
        cb.ax.tick_params(which='minor', right=False)



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
