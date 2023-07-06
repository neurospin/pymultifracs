import matplotlib.colors as colors
import matplotlib.ticker as ticker

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid", context='notebook')
figure_folder = 'figures/fBm/'

from pymultifracs.psd import plot_psd, PSD
from pymultifracs.viz import plot_coef
from pymultifracs.simul import fbm, mrw
from pymultifracs.mfa import mf_analysis
from pymultifracs.wavelet import wavelet_analysis

import numpy as np

from pymultifracs.simul.noisy import create_oscillation, generate_osc_freq,\
    create_mask3


N = int(1.5 * 2 ** 17)
X = fbm(shape=N, H=0.8)
X_freq_drift = fbm(shape=N, H=0.3)
X_mrw = mrw(shape=N, H=0.8, lam=np.sqrt(0.03), L=N)

X_diff = np.diff(X)
X_mrw_diff = np.diff(X_mrw)

rng = np.random.default_rng(seed=42)

mask = create_mask3(X_diff.shape[0], 3, (X_diff.shape[0] + 1) // 100, 12)
noise = fbm(shape=X.shape[0], H=.9)
noise = mask * np.diff(noise)
noisy_X = 10 * noise + X_diff

WT = wavelet_analysis(noisy_X, p_exp=2, j1=8, j2_reg=12, gamint=1)

from pymultifracs.cumulants import qn_scale2
from pymultifracs._robust import limits

input = np.log(abs(WT.wt_coefs.values[1]))
input = input[~np.isnan(input)]

# limits()
# print(input.shape)
# Q = qn_scale2(input[:1000])
# print(Q)

dwt, lwt = mf_analysis([WT.wt_coefs, WT.wt_leaders],
                       scaling_ranges=[(4, WT.wt_leaders.j2_eff() - 2)],
                       robust=True)

print(lwt.cumulants.c2)
