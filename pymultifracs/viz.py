import time
import os
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.signal import welch

# from .wavelet import _estimate_eta_p, wavelet_analysis
from . import wavelet, multiresquantity


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
              C_fmt='--.', lw_C=None, offset=0, plot_CI=True, signal_idx1=0,
              signal_idx2=0, **C_kwargs):
    
    if cm.mode == 'pairwise':
        signal_idx2 = 0

    j1, j2, j_min, j_max = cm.get_jrange(j1, j2)

    if cm.j.min() > j1:
        raise ValueError(f"Expected mrq to have minium scale {j1=}, got "
                         f"{cm.j.min()} instead")

    m1 = cm.m[ind_m1]
    m2 = cm.m[ind_m2]

    idx = np.s_[j_min:j_max]

    x = cm.j[idx]
    y = getattr(cm, f'C{m1}{m2}')[
        idx, scaling_range, signal_idx1, signal_idx2, 0]

    if cm.bootstrapped_obj is not None and plot_CI:

        if cm.bootstrapped_obj.j.min() > j1:
            raise ValueError(
                f"Expected bootstrapped mrq to have minimum scale {j1=}, got "
                f"{cm.bootstrapped_obj.j.min()} instead")

        CI = getattr(cm, f'CIE_C{m1}{m2}')[
            j1 - cm.bootstrapped_obj.j.min():
            j2 - cm.bootstrapped_obj.j.min() + 1]

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
    ax.set_ylabel(f'$C_{{{m1}{m2}}}(j)$')
    # ax.grid()
    # plt.draw()

    if len(cm.log_cumulants) > 0 and plot_fit:

        x0, x1 = cm.scaling_ranges[scaling_range]

        slope_log2_e = getattr(cm, f'c{m1}{m2}')[
            scaling_range, signal_idx1, signal_idx2, 0]
        slope = slope_log2_e / np.log2(np.e)
        
        intercept = cm.intercept[ind_m1, ind_m2, scaling_range].reshape(
            *cm.values.shape[4:]
        )[signal_idx1, signal_idx2, 0]

        y0 = slope*x0 + intercept
        y1 = slope*x1 + intercept

        if cm.bootstrapped_obj is not None:
            CI = getattr(cm, f"CIE_c{m1}{m2}")[
                scaling_range, signal_idx1, signal_idx2]
            CI_legend = (
                f"; [{cp_string_format(CI[scaling_range, 1], True)}, "
                f"{cp_string_format(CI[scaling_range, 0], True)}]")
        else:
            CI_legend = ""

        legend = (rf'$c_{{{m1}{m2}}}$ = {cp_string_format(slope_log2_e)}'
                  + CI_legend)

        ax.plot([x0, x1], [y0, y1], color=fit_color,
                linestyle='-', linewidth=lw_fit, label=legend, zorder=2)
        if plot_legend:
            ax.legend()

    ax.set(xlim=(j1-.5, j2+.5))


def plot_cm(cm, ind_m, j1, j2, scaling_range, ax, C_color='grey',
            fit_color='k', plot_legend=False, lw_fit=2, plot_fit=True,
            C_fmt='--.', lw_C=None, offset=0, plot_CI=True, signal_idx=0,
            shift_gamint=False, **C_kwargs):

    j1, j2, j_min, j_max = cm.get_jrange(j1, j2, plot_CI)

    m = cm.m[ind_m]

    x = cm.j[j_min:j_max]

    y = getattr(cm, f'C{m}')[j_min:j_max, scaling_range, signal_idx, 0]

    if shift_gamint and ind_m == 0:
        y -= x * cm.gamint / np.log2(np.e)

    if cm.bootstrapped_obj is not None and plot_CI:

        if cm.bootstrapped_obj.j.min() > j1:
            raise ValueError(
                f"Expected bootstrapped mrq to have minimum scale {j1=}, got "
                f"{cm.bootstrapped_obj.j.min()} instead")

        CI_slice = np.s_[int(j1 - cm.bootstrapped_obj.j.min()):
                         int(j2 - cm.bootstrapped_obj.j.min() + 1)]

        CI = getattr(cm, f'CIE_C{m}')[CI_slice, scaling_range, signal_idx]

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

    ax.set(xlabel='Temporal scale $j$', ylabel=f'$C_{m}(j)$')

    if len(cm.log_cumulants) > 0 and plot_fit:

        x0, x1 = cm.scaling_ranges[scaling_range]
        slope_log2_e = cm.log_cumulants[ind_m, scaling_range, signal_idx]

        if shift_gamint and ind_m == 0:
            slope_log2_e -= cm.gamint

        slope = slope_log2_e / np.log2(np.e)
        # slope = cm.slope[ind_m, scaling_range, signal_idx]

        intercept = cm.intercept[ind_m, scaling_range, signal_idx]

        y0 = slope*x0 + intercept + offset
        y1 = slope*x1 + intercept + offset

        if cm.bootstrapped_obj is not None:
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

        ax.tick_params(top=False, right=False, which='minor')

        ax.set(xlim=(j1-.5, j2+.5))


def plot_cumulants(cm, figsize, nrow=2, j1=None, filename=None,
                   scaling_range=0, legend=True, n_cumul=None, signal_idx=0,
                   fignum=1, **kw):
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
        plot_dim_2 = int(np.ceil(n_cumul / nrow))
    else:
        plot_dim_1 = 1
        plot_dim_2 = 1

    if figsize is None:
        figsize = (3.3 * plot_dim_2, 1 * plot_dim_1)

    fig, axes = plt.subplots(plot_dim_1,
                             plot_dim_2,
                             num=fignum,
                             squeeze=False,
                             figsize=figsize,
                             sharex=True)

    for ind_m, m in enumerate(cm.m[:n_cumul]):

        ax = axes[ind_m % nrow][ind_m // nrow]

        plot_cm(cm, ind_m, j1, None, scaling_range, ax, plot_legend=legend,
                signal_idx=signal_idx, **kw)
        
    for j in range(ind_m):

        if j % nrow == nrow-1:
            continue

        axes[j % nrow][j // nrow].xaxis.set_visible(False)

    for j in range(ind_m + 1, len(axes.flat)):
        fig.delaxes(axes[j % nrow][j // nrow])

    if filename is not None:
        plt.savefig(filename)


def plot_coef(mrq, j1, j2, ax=None, vmin=None, vmax=None, cbar=True,
              figsize=(2.5, 1), gamma=.3, nan_idx=None, signal_idx=0,
              cbar_kw=None, cmap='magma'):

    leader = isinstance(mrq, multiresquantity.WaveletLeader)
    leader_idx_correction = True

    if vmax is None:
        max_scale = [
            np.nanmax(mrq.values[s][:, signal_idx])
            for s in range(j1, j2+1) if s in mrq.values
        ]

    if vmin is None:
        min_scale = [
            np.nanmin(np.abs(mrq.values[s][:, signal_idx]))
            for s in range(j1, j2+1) if s in mrq.values
        ]

    if leader and not np.isinf(mrq.p_exp):

        if mrq.eta_p is None:

            mrq.eta_p = wavelet._estimate_eta_p(
                mrq.origin_mrq, mrq.p_exp, [(j1, j2)], False, None
            )

        ZPJCorr = mrq.correct_pleaders(j1, j2)[None, 0, signal_idx]

        if vmax is None:
            max_scale = [
                m * ZPJCorr[:, scale-j1] 
                for m, scale in zip(max_scale, range(j1, j2+1))
            ]

        if vmin is None:
            min_scale = [
                m * ZPJCorr[:, scale-j1] 
                for m, scale in zip(min_scale, range(j1, j2+1))
            ]

    if vmax is None:
        vmax = max(max_scale)
    if vmin is None:
        vmin = min(min_scale)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')#, width_ratios=[20, 1])

    norm = PowerNorm(vmin=vmin, vmax=vmax, gamma=gamma)
    if isinstance(cmap, str):
        cmap = sns.color_palette(cmap, as_cmap=True)
    cmap.set_bad('grey')

    for i, scale in enumerate(range(j1, j2 + 1)):

        if scale not in mrq.values:
            continue

        temp = mrq.get_values(scale)[:, 0, signal_idx]

        X = ((np.arange(temp.shape[0] + 1)
            #   + (1 if leader and leader_idx_correction else 0))
            #   + 1
              ) * (2 ** (scale - j1)))
        # X += scale - 1
        # if leader and scale > 1:
        #     X += 2 ** (scale - j1 - 1)
        X = np.tile(X[:, None], (1, 2))

        C = np.copy(temp[:, None])

        if not leader:
            C = np.abs(C)

        # Correcting potential p_leaders
        # if leader and not np.isinf(mrq.p_exp):
        #     C *= ZPJCorr[:, scale - j1]

        Y = np.ones(X.shape[0]) * scale
        Y = np.stack([Y - .5, Y + .5]).transpose()

        qm = ax.pcolormesh(X, Y, C, cmap=cmap, norm=norm, rasterized=True)

        if nan_idx is not None and scale in nan_idx:
            idx = np.unique(np.r_[nan_idx[scale], nan_idx[scale] + 1])

            segments = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)

            for seg in segments:

                if len(seg) == 0:
                    continue

                ax.pcolormesh(
                    X[seg[[0, -1]]], Y[seg[[0, -1]]], C[[0]], alpha=1,
                    edgecolor='xkcd:blue')

    ax.set(ylim=(j1-.5, j2+.5), yticks=range(j1, j2+1),
           xlabel='shift $k$', ylabel='scale $j$', facecolor='grey',
           xlim=(0, mrq.values[j1].shape[0]))

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
        # 

        if cbar_kw is None:
            cbar_kw = dict(
                fraction=.1, aspect=8,
                ticks=mpl.ticker.MaxNLocator(4, symmetric=False))

        cb = plt.colorbar(qm, ax=ax, **cbar_kw)
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


PSD = namedtuple('PSD', 'freq psd')
"""Aggregates power spectral density information

Attributes
----------
freq : ndarray
    Frequency support of the psd values
psd : ndarray
    Power density associated matching the frequencies in ``freq``
"""


def plot_psd(signal, fs, n_fft=4096, seg_size=None, n_moments=2,
             log='log2', ax=None, wt='db'):
    """
    Plot the superposition of Fourier-based Welch estimation and Wavelet-based
    estimation of PSD on a log-log graphic.

    Based on the `wavelet_fourier_spectrum` function of the MATLAB toolbox
    mf_bs_toolbox-v0.2 [1]_

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Time series of sampled values

    fs : float
        Sampling frequency of ``signal``

    n_fft : int, optional
        Length of the FFT desired.
        If `seg_size` is greater, ``n_fft = seg_size``.

    seg_size : int | None
        Length of Welch segments.
        Defaults to None, which sets it equal to ``n_fft``

    n_moments : int
        Number of vanishing moments of the Daubechies wavelet used in the
        Wavelet decomposition.

    log : str
        Log function to use on the frequency and power axes

    References
    ----------
    .. [1] \
        http://www.ens-lyon.fr/PHYSIQUE/Equipe3/Multifractal/dat/mf_bs_tool-v0.2.zip
    """

    # Computing

    freq_fourier, psd_fourier = welch_estimation(signal, fs, n_fft,
                                                 seg_size)
    freq_wavelet, psd_wavelet = wavelet_estimation(signal, fs, n_moments, wt=wt)

    # Plotting

    freq = [freq_fourier, freq_wavelet]
    psd = [psd_fourier, psd_wavelet]
    legend = ['Fourier', 'Wavelet']
    log_plot(freq, psd, legend, log=log, ax=ax)


log_function = {'log2': np.log2,
                'log': np.log}


def _log_psd(freq, psd, log):
    """
    Compute the logged values of a PSD and its frequency support similarly to
    the MATLAB toolbox
    """

    # Avoid computing log(0)
    if np.any(freq == 0):
        support = [freq != 0.0][0]
        freq, psd = freq[support], psd[support]

    # Compute the logged values of the frequency and psd
    log = log_function[log]
    freq, psd = log(freq), log(psd)

    return freq, psd


def log_plot(freq_list, psd_list, legend=None, fmt=None, color=None, slope=[],
             log='log2', lowpass_freq=np.inf, xticks=None,
             title='Power Spectral Density', ax=None, show=False,
             plot_kwargs={}, linewidth=None):
    """
    Perform a log-log plot over a list of paired frequency range and PSD, with
    optional legend and fitted slope

    Parameters
    ----------
    freq_list: list
        list of frequency supports of the PSDs to plot

    psd_list: list
        list of PSDs to plot

    legend: list | None
        list of labels to assign to the PSDs

    color: list | None
        colors to assign to the plotted PSDs

    slope: [(freq, psd)] | []
        list of 2-tuples containing the frequency support and PSD
        representation of a slope to plot
        TODO: replace (freq, psd) with (beta, log_C)

    log: str
        name of log function to use on the data before plotting
    """

    ax = plt.gca() if ax is None else ax

    ax.set_xlabel(f'{log} f')
    ax.set_ylabel(f'{log} S_2(f)')
    ax.set_title(title)

    if color is None:
        cmap = plt.get_cmap("tab10")
        color = [cmap(i % 10) for i in range(len(freq_list))]

    if linewidth is None:
        linewidth = [1.5] * len(freq_list)

    if fmt is None:
        fmt = ['-'] * len(freq_list)

    for i, (freq, psd, f, col, lw) in enumerate(zip(freq_list, psd_list, fmt,
                                                    color, linewidth)):

        indx = tuple([freq < lowpass_freq])
        freq, psd = freq[indx], psd[indx]
        log_freq, psd = _log_psd(freq, psd, log)  # Log frequency and psd

        ax.plot(log_freq, psd, f, c=col, lw=lw, **plot_kwargs)

        if xticks is not None and i == len(freq_list) - 1:
            ax.set_xticks(log_freq)
            ax.set_xticklabels([f'{fr:.2f}' if fr < 1 else f'{fr:.1f}'
                                for fr in freq])
            # plt.xticks(log_freq, [f'{fr:.2f}' for fr in freq])

    if color is None:
        c_gen = (f'C{i}' for i in range(len(freq_list)))
    else:
        c_gen = color

    for tup, c in zip(slope, c_gen):
        ax.plot(*tup, color=c, dashes=[6, 2], lw=2)

    if legend is not None:
        ax.legend(legend)

    if show:
        plt.show()


def welch_estimation(signal, fs, n_fft=4096, seg_size=None):
    """
    Wrapper for :obj:`scipy.signal.welch`

    Parameters
    ----------
    signal : 1D-array_like
        Time series of sampled values

    fs : float
        Sampling frequency of the signal

    n_fft : int, optional
        Length of the FFT desired.
        If `seg_size` is greater, ``n_fft = seg_size``.

    seg_size : int | None
        Length of Welch segments.
        Defaults to None, which sets it equal to `n_fft`

    Returns
    -------
    psd : PSD
    """

    # Input argument sanitizing

    if seg_size is None:
        seg_size = n_fft

    if n_fft < seg_size:
        n_fft = seg_size

    # Frequency
    freq = fs * np.linspace(0, 0.5, n_fft // 2 + 1)

    # PSD
    _, psd = welch(signal,
                   window='hamming',
                   nperseg=seg_size,
                   noverlap=seg_size / 2,
                   nfft=n_fft,
                   detrend=False,
                   return_onesided=True,
                   scaling='density',
                   average='mean',
                   fs=2 * np.pi)

    psd *= 4        # compensating for negative frequencies
    psd = np.array(psd)

    return PSD(freq=freq, psd=psd)


def wavelet_estimation(signal, fs, n_moments, j2=None, wt='db'):
    """
    PSD estimation based on Wavelet coefficients

    The PSD is estimated using :obj:`~pymultifracs.wavelet.wavelet_analysis`.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Time series
    fs : float
        Sampling frequency of the signal
    n_moments : int
        Number of vanishing moments of the Daubechies wavelet used in the
        wavelet transform.

    Returns
    -------
    psd : PSD
    """

    # PSD
    transform = wavelet.wavelet_analysis(signal, j2=j2,
                                         normalization=1,
                                         wt_name=f'{wt}{n_moments}',
                                         gamint=0.5,
                                         p_exp=None)

    # for arr in transform.wt_coefs.values.values():

    psd = [np.nanmean(np.square(arr), axis=0)
           for arr in transform.wt_coefs.values.values()]
    psd = np.array(psd)

    # Frequency
    scale = np.arange(len(psd)) + 1
    freq = (3/4 * fs) / (np.power(2, scale))

    return PSD(freq=freq, psd=psd)
