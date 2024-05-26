from scipy.stats import norm
import torch


class VarCov:
    def __init__(self, alpha) -> None:
        """
        Initializes the VarCov class with a specified confidence level.

        Parameters
        ----------
        alpha : float
            Confidence level for Value at Risk (VaR) and Expected Shortfall (ES) calculations.
        """
        self.alpha = alpha

    def fit(self, *args, **kwargs):
        """
        Placeholder fit method for compatibility with other models.
        """
        pass

    def predict(self, context, **kwargs):
        """
        Predicts Value at Risk (VaR) and Expected Shortfall (ES) for univariate context.

        Parameters
        ----------
        context : torch.Tensor
            Input data tensor.
        **kwargs : dict
            Additional arguments.

        Returns
        -------
        tuple
            VaR and ES values.
        """
        if context.shape[-1] > 1:
            return self.predict_multivariate(context, **kwargs)
        context = context.flatten()
        mu, sigma = norm.fit(context)
        VaR = mu + sigma * norm.ppf(self.alpha)
        ES = -1 * (mu + sigma * norm.pdf(norm.ppf(self.alpha)) / self.alpha)
        return VaR, ES

    def predict_multivariate(self, context, **kwargs):
        """
        Predicts multivariate VaR and ES based on individual VaR and ES aggregated along assets in a portfolio.

        Parameters
        ----------
        context : torch.Tensor
            TxL array, where L is the number of variables.
        **kwargs : dict
            Additional arguments, which may include:
            - scaler : Scaler object for inverse transforming the context.

        Returns
        -------
        tuple
            VaR and ES values.
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
            + torch.sqrt(w.T @ sigma @ w) * norm.pdf(norm.ppf(self.alpha)) / self.alpha
        )
        return VaR, ES
