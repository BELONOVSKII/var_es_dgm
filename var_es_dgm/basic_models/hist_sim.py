import torch
import numpy as np


class HistoricalSimulation:
    def __init__(self, alpha=0.05) -> None:
        self.alpha = alpha

    def fit(self, *args, **kwargs):
        pass

    def predict(self, context, **kwargs):
        if context.shape[-1] > 1:
            return self.predict_multivariate(context, **kwargs)
        context = context.flatten()
        VaR = torch.quantile(context, q=self.alpha)
        ES = context[torch.where(context <= VaR)[0]]
        ES = torch.sum(ES) / ES.shape[0]
        return VaR, ES

    def predict_multivariate(self, context, **kwargs):
        """Predicts mutlivariate VaR and ES based on individual VaR, ES aggreagated along assets in portfolio

        Args:
            context: TxL array, where L is the number of variables
        """
        if "scaler" in kwargs:
            context = torch.tensor(
                kwargs["scaler"].inverse_transform(torch.squeeze(context))
            )

        # estimating individual VaR and ES
        n_vars = context.shape[-1]
        VaRs = torch.quantile(context, q=self.alpha, dim=0) / n_vars

        ESs = torch.zeros(n_vars)
        for i in range(n_vars):
            context_i = context[:, i]
            VaR_i = VaRs[i]
            ES_i = context_i[torch.where(context_i <= VaR_i)[0]]
            ESs[i] = torch.sum(ES_i) / (ES_i.shape[0] * n_vars)

        # estimating correlation matrix for VaR or taking provided
        if "R" in kwargs:
            R = kwargs["R"]
        else:
            R = torch.corrcoef(context.T)

        # computing final VaR and ES
        VaR = -torch.sqrt(VaRs @ R @ VaRs.T)
        ES = torch.sum(ESs)

        return VaR, ES
