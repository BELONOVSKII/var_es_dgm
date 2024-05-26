from scipy.stats import chi2
import numpy as np
import torch


def KupicksPOF(real, VaR, alpha=0.05, statistic=True):
    diff = real - VaR

    M = diff[diff < 0].shape[0]
    T = real.shape[0]

    num = (1 - alpha) ** (T - M) * alpha ** (M)
    den = (1 - M / T) ** (T - M) * (M / T) ** M

    LR_POF = -2 * np.log(num / den)

    if statistic:
        return chi2.sf(LR_POF, df=1), LR_POF

    return chi2.sf(LR_POF, df=1)


def HaasTBF(real, VaR, alpha=0.05, statistic=True):
    diff = real - VaR
    idx = torch.where(diff < 0)[0]
    N = idx[1:] - idx[:-1]

    S = 0
    for N_i in N:
        num = (1 - alpha) ** (N_i - 1) * (alpha)
        den = 1 / N_i * (1 - 1 / N_i) ** (N_i - 1)
        S += np.log(num / den)
    LR_TUFF = -2 * S

    if statistic:
        return chi2.sf(LR_TUFF, df=idx.shape[0] - 1), LR_TUFF

    return chi2.sf(LR_TUFF, df=idx.shape[0] - 1)
