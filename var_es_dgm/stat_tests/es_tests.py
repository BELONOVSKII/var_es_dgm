import torch


def ES_G(real, ES):
    y = torch.sort(real - ES)[0]
    y = torch.cumsum(y, dim=0)
    G_n = torch.where(y < 0)[0].shape[0] / real.shape[0]
    return G_n


def AcerbiSze1(real, VaR, ES):
    exceedens = torch.where(real <= VaR)[0]
    Z1 = torch.sum((real[exceedens] / ES[exceedens] + 1)) / exceedens.shape[0]
    return Z1.item()


def AcerbiSze2(real, VaR, ES, alpha):
    exceedens = torch.where(real <= VaR)[0]
    Z2 = torch.sum((real[exceedens] / ES[exceedens] + 1)) / real.shape[0] * alpha
    return Z2.item()
