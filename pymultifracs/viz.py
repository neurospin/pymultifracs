import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from .psd import log_plot


def plot_multiscale(results, seg2color, ax=None):

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
        seg: (freq := results.loc[subjects[0], seg].freq)[(freq <= maxf[seg])
                                                          & (freq >= minf[seg])]
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

    ticks = ax.xticks()[0]
    labels = [f'{t:g}\n{2 ** t:.2g}' for t in ticks]

    ax.xticks(ticks, labels)

    sns.despine()

    ax.xlabel('log2 f', fontsize=18, c='black')
    ylim = ax.ylim()
    ax.vlines(x=[results.iloc[0].slope[0][0], results.iloc[0].slope[0][-1]],
              ymin=ylim[0], ymax=ylim[1], linestyles='dashed',
              colors='#0000003f')
    ax.set_ylim(ylim)
