from scipy.stats import norm
import torch


class VarCov:
    def __init__(self, alpha) -> None:
        self.alpha = alpha

    def fit(self, data):
        pass

    def predict(self, context, **kwargs):
        if context.shape[-1] > 1:
            return self.predict_multivariate(context, **kwargs)
        context = context.flatten()
        mu, sigma = norm.fit(context)
        VaR = mu + sigma * norm.ppf(self.alpha)
        ES = -1 * (mu + sigma * norm.pdf(norm.ppf(self.alpha)) / (self.alpha))
        return VaR, ES

    def predict_multivariate(self, context, **kwargs):
        """Predicts mutlivariate VaR and ES based on individual VaR, ES aggreagated along assets in portfolio

        Args:
            context: TxL array, where L is the number of variables
        """
        context = torch.tensor(
            kwargs["scaler"].inverse_transform(torch.squeeze(context))
        )
        n_vars = context.shape[-1]
        mu = torch.mean(context, dim=0).to(torch.float)
        w = torch.tensor([1 / n_vars for _ in range(n_vars)]).to(torch.float)
        sigma = torch.cov(context.T).to(torch.float)
        VaR = w.T @ mu + torch.sqrt(w.T @ sigma @ w) * norm.ppf(0.05)
        ES = -1 * (
            w.T @ mu
            + torch.sqrt(w.T @ sigma @ w)
            * norm.pdf(norm.ppf(self.alpha))
            / (self.alpha)
        )
        return VaR, ES
