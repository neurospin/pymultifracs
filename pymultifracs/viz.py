"""
Authors: Merlin Dumeur <merlin@dumeur.net>
"""


import time
import os
from collections import namedtuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import seaborn as sns
import numpy as np
from scipy.signal import welch

# from .wavelet import estimate_eta_p, wavelet_analysis
from . import wavelet, multiresquantity, estimation
from .utils import Dim


def _cp_string_format(cp, CI=False):
    """
    Formats :math:`c_p` coefficients' values.
    """

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


def _get_CI_legend(CI):

    return (
        f"; [{_cp_string_format(CI.loc[{Dim.CI: 'lower'}], True)}, "
        f"{_cp_string_format(CI.loc[{Dim.CI: 'upper'}], True)}]"
    )


def plot_bicm(cm, m1, m2, j1, j2, scaling_range, ax, C_color='grey',
              fit_color='k', plot_legend=False, lw_fit=2, plot_fit=True,
              C_fmt='--.', lw_C=None, offset=0, plot_CI=True, signal_idx1=0,
              signal_idx2=0, **C_kwargs):
    """
    Plots bivariate cumulants.
    """

    if cm.mode == 'pairwise':
        signal_idx2 = 0

    j1, j2, j_min, j_max = cm.get_jrange(j1, j2)

    if cm.j.min() > j1:
        raise ValueError(f"Expected mrq to have minium scale {j1=}, got "
                         f"{cm.j.min()} instead")

    # ind_m = cm.m.index((m1, m2))
    idx = np.s_[j_min:j_max]

    x = cm.j[idx]
    y = getattr(cm, f'C{m1}{m2}')

    if m1 == 0 and m2 > 0:
        y = y.sel(channel2=signal_idx2)
    elif m2 == 0 and m1 > 0:
        y = y.sel(channel1=signal_idx1)
    else:
        y = y.sel(channel1=signal_idx1, channel2=signal_idx2)

    if cm.bootstrapped_obj is not None and plot_CI:

        if cm.bootstrapped_obj.j.min() > j1:
            raise ValueError(
                f"Expected bootstrapped mrq to have minimum scale {j1=}, got "
                f"{cm.bootstrapped_obj.j.min()} instead")

        CI = getattr(cm, f'CIE_C{m1}{m2}').sel(
            j=slice(j1, j2))

        CI -= y
        CI.loc[{Dim.CI, 'lower'}] *= -1
        assert (CI < 0).sum() == 0
        CI = CI.transpose(Dim.CI, Dim.j)

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

    if len(cm.log_cumulants) > 0 and plot_fit:

        x0, x1 = cm.scaling_ranges[scaling_range]

        slope_log2_e = getattr(cm, f'c{m1}{m2}').isel(
            scaling_range=scaling_range, channel1=signal_idx1,
            channel2=signal_idx2)
        slope = slope_log2_e / np.log2(np.e)

        # match m1, m2:
        #     case 0, m2:
        #         intercept = cm.margin2_intercept[ind_m2, scaling_range].

        intercept = cm.intercept.sel(
            m1=m1, m2=m2).isel(
                scaling_range=scaling_range, channel1=signal_idx1,
                channel2=signal_idx2)

        y0 = slope*x0 + intercept
        y1 = slope*x1 + intercept

        if cm.bootstrapped_obj is not None:
            CI = getattr(cm, f"CIE_c{m1}{m2}").isel(
                scaling_range=scaling_range, channel1=signal_idx1,
                channel2=signal_idx2)
            CI_legend = _get_CI_legend(CI)
        else:
            CI_legend = ""

        legend = (rf'$c_{{{m1}{m2}}}$ = {_cp_string_format(slope_log2_e)}'
                  + CI_legend)

        ax.plot([x0, x1], [y0, y1], color=fit_color,
                linestyle='-', linewidth=lw_fit, label=legend, zorder=2)
        if plot_legend:
            ax.legend()

    ax.set(xlim=(j1-.5, j2+.5))


def plot_cm(cm, ind_m, j1, j2, range_idx, ax, C_color='grey',
            fit_color='k', plot_legend=False, lw_fit=2, plot_fit=True,
            C_fmt='--.', lw_C=None, offset=0, plot_CI=True, signal_idx=0,
            shift_gamint=False, **C_kwargs):
    """
    Helper function to plot individual :math:`C_m(j)` functions along with
    their associated :math:`c_m` fit.

    Parameters
    ----------
    cm : :class:`.Cumulants`
        Cumulants to plot.
    ind_m : int
        Index of the cumulant order :math:`m` to plot.
        For example if ``cm.m = [1, 2, 3]`` then ``ind_m=2`` plots
        math:`C_2(j)`.
    j1 : int
        Lower limit of temporal scales to plot.
    j2 : int
        Upper limit of temporal scales to plot.
    range_idx : int
        If multiple scaling ranges were used in fitting, indicates the
        index to use.
    ax : :class:`~matplotlib.axes.Axes`
        Mandatory argument: axes on which to plot the function.
    C_color : str
        Color for the :math:`C_m(j)` function plot.
    fit_color : str
        Color for the :math:`c_m` regression plot.
    plot_legend : bool
        If true, displays legend for the :math:`c_m` with estimated fit.
    lw_fit : int
        Linewidth of the :math:`c_m` regression.
    plot_fit : True
        If False, the :math:`c_m` fit is not plotted. Defaults to True.
    C_fmt : str
        Formatting string for the :math:`C_m(j)` plot.
    lw_C : int | None
        Linewidth of the :math:`C_m(j)` plot.
    offset : int
        y-axis offset for the plot, useful when showing multiple signals at
        once.
    plot_CI : bool
        If bootstrapping was used, show bootstrap-derived confidence intervals.
    signal_idx : int
        If using a multivariate signal, index of the signal to plot.
    shift_gamint : bool
        If fractional integration was used, shifts the :math:`C_1(j)` plot by
        :math:`-j \\gamma / \\log_2(e)`, and adjusts the :math:`c_1` fit
        accordingly.
    **C_kwargs : dict
        Additional arguments are passed to the plotting function for
        :math:`C_m(j)`
    """

    j1, j2, j_min, j_max = cm.get_jrange(j1, j2, plot_CI)

    m = cm.m[ind_m]

    x = cm.j[j_min:j_max]

    y = getattr(cm, f'C{m}').sel(
        j=slice(j1, j2)).isel(scaling_range=range_idx, channel=signal_idx)

    if shift_gamint and ind_m == 0:
        y -= x * cm.gamint / np.log2(np.e)

    if cm.bootstrapped_obj is not None and plot_CI:

        if cm.bootstrapped_obj.j.min() > j1:
            raise ValueError(
                f"Expected bootstrapped mrq to have minimum scale {j1=}, got "
                f"{cm.bootstrapped_obj.j.min()} instead")

        CI_slice = np.s_[int(j1 - cm.bootstrapped_obj.j.min()):
                         int(j2 - cm.bootstrapped_obj.j.min() + 1)]

        CI = getattr(cm, f'CIE_C{m}').sel(
            j=slice(j1, j2)).isel(scaling_range=range_idx, channel=signal_idx)

        CI -= y
        CI.loc[{Dim.CI: 'lower'}] *= -1
        assert (CI < 0).sum() == 0
        # CI[CI < 0] = 0
        CI = CI.transpose(Dim.CI, Dim.j)

    else:
        CI = None

    y += offset

    errobar_params = {
        'zorder': -1
    }

    errobar_params.update(C_kwargs)

    ax.errorbar(x, y, CI, fmt=C_fmt, color=C_color, lw=lw_C, **errobar_params)

    ax.set(xlabel='Temporal scale $j$',
           ylabel=f'$C_{m}{cm.variable_suffix}(j)$')

    if len(cm.log_cumulants) > 0 and plot_fit:

        x0, x1 = cm.scaling_ranges[range_idx]
        slope_log2_e = cm.log_cumulants.sel(m=m).isel(
            scaling_range=range_idx, channel=signal_idx)

        if shift_gamint and ind_m == 0:
            slope_log2_e -= cm.gamint

        slope = slope_log2_e / np.log2(np.e)
        # slope = cm.slope[ind_m, scaling_range, signal_idx]

        intercept = cm.intercept.sel(m=m).isel(
            scaling_range=range_idx, channel=signal_idx)

        y0 = slope*x0 + intercept + offset
        y1 = slope*x1 + intercept + offset

        if cm.bootstrapped_obj is not None and plot_CI:
            CI = getattr(cm, f"CIE_c{m}")[range_idx, signal_idx]

            CI_legend = _get_CI_legend(CI)
        else:
            CI_legend = ""

        legend = (
            rf'$c_{m}{cm.variable_suffix}$ = {_cp_string_format(slope_log2_e)}'
            + CI_legend)

        ax.plot([x0, x1], [y0, y1], color=fit_color,
                linestyle='-', linewidth=lw_fit, label=legend, zorder=0)
        if plot_legend:
            ax.legend()

        ax.tick_params(top=False, right=False, which='minor')

        ax.set(xlim=(j1-.5, j2+.5))


def plot_cumulants(cm, figsize, nrow=2, j1=None, j2=None, filename=None,
                   range_idx=0, legend=True, n_cumul=None, signal_idx=0,
                   **kw):
    """
    Plots cumulants from a :class:`.Cumulants` object.
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
        figsize = (3.5 * plot_dim_2, 1.35 * plot_dim_1)

    fig, axes = plt.subplots(plot_dim_1,
                             plot_dim_2,
                             squeeze=False,
                             figsize=figsize,
                             sharex=True,
                             layout='tight')

    # for ind_m, m in enumerate(cm.m[:n_cumul]):
    for ind_m in range(n_cumul):

        ax = axes[ind_m % nrow][ind_m // nrow]

        plot_cm(
            cm=cm, ind_m=ind_m, j1=j1, j2=j2, range_idx=range_idx,
            ax=ax, plot_legend=legend, signal_idx=signal_idx, **kw)

    for j in range(ind_m):

        if j % nrow == nrow-1:
            continue

        axes[j % nrow][j // nrow].xaxis.set_visible(False)

    for j in range(ind_m + 1, len(axes.flat)):
        fig.delaxes(axes[j % nrow][j // nrow])

    if filename is not None:
        plt.savefig(filename)


def plot_coef(mrq, j1, j2, ax=None, vmin=None, vmax=None, cbar=True,
              figsize=(2.5, 1), gamma=1, nan_idx=None, signal_idx=0,
              cbar_kw=None, cmap='magma'):
    """
    Plots the coefficients from a MRQ.
    """

    leader = isinstance(mrq, multiresquantity.WaveletLeader)
    # leader_idx_correction = True

    # if vmax is None:
    #     max_scale = [
    #         np.nanmax(mrq.values[s][:, signal_idx])
    #         for s in range(j1, j2+1) if s in mrq.values
    #     ]

    # if vmin is None:
    #     min_scale = [
    #         np.nanmin(np.abs(mrq.values[s][:, signal_idx]))
    #         for s in range(j1, j2+1) if s in mrq.values
    #     ]

    values = [mrq.get_values(scale).isel(channel=signal_idx)
              for scale in range(j1, j2+1)]

    # if leader and not np.isinf(mrq.p_exp):

    #     if mrq.eta_p is None:

    #         mrq.eta_p = wavelet._estimate_eta_p(
    #             mrq.origin_mrq, mrq.p_exp, [(j1, j2)], False, None
    #         )

    #     ZPJCorr = mrq._correct_pleaders()[None, 0, signal_idx]

    #     if vmax is None:
    #         max_scale = [
    #             m * ZPJCorr[:, scale-j1]
    #             for m, scale in zip(max_scale, range(j1, j2+1))
    #         ]

    #     if vmin is None:
    #         min_scale = [
    #             m * ZPJCorr[:, scale-j1]
    #             for m, scale in zip(min_scale, range(j1, j2+1))
    #         ]

    if vmax is None:
        vmax = max([np.nanmax(val.values) for val in values])
    if vmin is None:
        vmin = min([np.nanmin(abs(val.values)) for val in values])

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=figsize, layout='constrained')
        # , width_ratios=[20, 1])

    norm = PowerNorm(vmin=vmin, vmax=vmax, gamma=gamma)
    if isinstance(cmap, str):
        cmap = sns.color_palette(cmap, as_cmap=True)
    cmap.set_bad('grey')

    # for i, scale in enumerate(range(j1, j2 + 1)):
    for scale in range(j1, j2 + 1):

        if scale not in mrq.values:
            continue

        temp = mrq.get_values(scale).isel(channel=signal_idx).transpose(
            Dim.k_j, ...)

        if Dim.scaling_range in temp.dims:
            temp = temp.isel(scaling_range=0)
        # temp = values[i]

        X = ((np.arange(temp.sizes[Dim.k_j] + 1)
              ) * (2 ** (scale - j1)))
        # X += scale - 1
        # if leader and scale > 1:
        #     X += 2 ** (scale - j1 - 1)
        X = np.tile(X[:, None], (1, 2))

        if not leader:
            C = np.abs(temp).values
        else:
            C = temp.values

        # Correcting potential p_leaders
        # if leader and not np.isinf(mrq.p_exp):
        #     C *= ZPJCorr[:, scale - j1]

        Y = np.ones(X.shape[0]) * scale
        Y = np.stack([Y - .5, Y + .5]).transpose()

        qm = ax.pcolormesh(X, Y, C[:, None], cmap=cmap, norm=norm, rasterized=True)

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
        # formatter = mpl.ticker.LogFormatterSciNotation(labelOnlyBase=False,
        #    minor_thresholds=(np.inf, np.inf))
        # formatter = mpl.ticker.LogFormatterSciNotation(labelOnlyBase=False,
        #    minor_thresholds=(np.inf, np.inf))

        if cbar_kw is None:
            cbar_kw = dict(
                fraction=.1, aspect=8,
                ticks=mpl.ticker.MaxNLocator(4, symmetric=False))

        if 'label' not in cbar_kw:
            cbar_kw['label'] = (
                f"${mrq._get_variable_name()}{mrq._get_suffix()[0]}(j, k)$")

        cb = plt.colorbar(qm, ax=ax, **cbar_kw)
        # plt.colorbar(qm, ax=ax    es[0], ticks=locator, aspect=1)
        cb.ax.tick_params(which='major', size=3)
        cb.ax.tick_params(which='minor', right=False)


# From pyvista, since https://github.com/pyvista/pyvista/issues/1125 is not yet
# fixed
# Start xvfb from Python.

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
    from pyvista import rcParams  # pylint: disable=C0415

    if os.name != 'posix':
        raise OSError('`start_xvfb` is only supported on Linux')

    if os.system('which Xvfb > /dev/null'):
        raise OSError(XVFB_INSTALL_NOTES)

    if window_size is None:
        window_size = rcParams['window_size']

    # use current default window size
    # pylint: disable=C0209
    window_size_parm = '%dx%dx24' % tuple(window_size)
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


def plot_psd(signal, fs, wt_name='db2', log_base=2, ax=None, **welch_kwargs):
    """
    Plot the superposition of Fourier-based Welch estimation and Wavelet-based
    estimation of PSD on a log-log graphic.

    Based on the `wavelet_fourier_spectrum` function of the MATLAB toolbox
    mf_bs_toolbox-v0.2 [1]_

    Parameters
    ----------
    signal : ndarray of float, shape (n_samples,)
        Time series of sampled values

    fs : float
        Sampling frequency of ``signal``

    wt_name : str
        Name of the decomposing wavelet.

    log_base : int
        Base of the logarithm in the plot axes.

    ax : :class:`~matplotlib.axes.Axes` | None
        Axes where to plot the PSD.

    **welch_kwargs : dict
        Arguments passed to the :func:`scipy.signal.welch` function.

    References
    ----------
    .. [1] \
        http://www.ens-lyon.fr/PHYSIQUE/Equipe3/Multifractal/dat/mf_bs_tool-v0.2.zip
    """

    if 'nperseg' not in welch_kwargs:
        welch_kwargs['nperseg'] = signal.shape[0] // 8

    # Computing

    freq_fourier, psd_fourier = welch_estimation(signal, fs, **welch_kwargs)
    freq_wavelet, psd_wavelet = wavelet_estimation(signal, fs, wt_name=wt_name)

    # Plotting

    freq_fourier = freq_fourier[1:]
    psd_fourier = psd_fourier[1:]

    freq = [freq_fourier, freq_wavelet]
    psd = [psd_fourier, psd_wavelet]
    legend = ['Fourier', 'Wavelet']
    log_plot(freq, psd, legend, log_base=log_base, ax=ax)


log_function = {'log2': np.log2,
                'log': np.log}


def log_plot(freq_list, psd_list, legend=None, fmt=None, color=None,
             slope=None, log_base=2, lowpass_freq=np.inf,
             title='Power Spectral Density', ax=None, show=False,
             linewidth=None, **plot_kwargs):
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

    slope: [(freq, psd)] | None
        list of 2-tuples containing the frequency support and PSD
        representation of a slope to plot
        TODO: replace (freq, psd) with (beta, log_C)

    log_base: int
        base of the logarithm in the x- and y-axes.
    """

    if slope is None:
        slope = []

    if ax is None:
        _, ax = plt.subplots()

    ax.set_xlabel('Frequency $f$ (Hz)')
    ax.set_ylabel('PSD')
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
        # log_freq, psd = _log_psd(freq, psd, log_base) # Log frequency and psd

        # idx_pos = (freq > 0) & (psd > 0)

        ax.loglog(freq, psd, f, base=log_base, c=col, lw=lw, **plot_kwargs)

        # if xticks is not None and i == len(freq_list) - 1:
        #     ax.set_xticks(log_freq)
        #     ax.set_xticklabels([f'{fr:.2f}' if fr < 1 else f'{fr:.1f}'
        #                         for fr in freq])
        #     plt.xticks(log_freq, [f'{fr:.2f}' for fr in freq])

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


def welch_estimation(signal, fs, **kwargs):
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
        If ``seg_size`` is greater, ``n_fft = seg_size``.

    seg_size : int | None
        Length of Welch segments.
        Defaults to None, which sets it equal to ``n_fft``

    Returns
    -------
    psd : PSD
    """

    # Input argument sanitizing

    # Frequency
    # freq = fs * np.linspace(0, 0.5, n_fft // 2 + 1)

    # PSD
    freq, psd = welch(signal,
                      window='hamming',
                      #   nperseg=seg_size,
                      #   noverlap=seg_size / 2,
                      #   nfft=n_fft,
                      detrend=False,
                      return_onesided=True,
                      scaling='density',
                      average='mean',
                      fs=fs,
                      **kwargs)

    psd = np.array(psd)

    return PSD(freq=freq, psd=psd)


def wavelet_estimation(signal, fs, j2=None, wt_name='db2'):
    """
    PSD estimation based on Wavelet coefficients

    The PSD is estimated using :obj:`~pymultifracs.wavelet.wavelet_analysis`.

    Parameters
    ----------
    signal : ndarray, shape (n_samples,)
        Time series.
    fs : float
        Sampling frequency of the signal.
    j2 : int | None
        Upper decomposition scale. Defaults to None, which selects the maximal
        available scale.
    wt_name : str
        Name of the decomposition wavelet. Defaults to Daubechies with 2 null
        moments.
    Returns
    -------
    psd : PSD
    """

    # PSD
    WT = wavelet.wavelet_analysis(
        signal, j2=j2, normalization=1, wt_name=wt_name)

    psd = [np.nanmean(np.square(arr[:, 0]), axis=0) for arr in WT.values.values()]
    psd = np.array(psd)

    # Frequency
    # scale = np.arange(len(psd)) + 1
    # freq = (3/4 * fs) / (np.power(2, scale))
    freq = WT.scale2freq(np.array([*WT.values]), fs)

    psd /= freq  # amplitude to density

    return PSD(freq=freq, psd=psd)
