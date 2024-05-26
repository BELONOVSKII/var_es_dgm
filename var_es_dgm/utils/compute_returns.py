from tqdm.auto import tqdm
import pandas as pd
import numpy as np


def compute_individual_returns(df):

    df_total = pd.DataFrame()
    df_total["Date"] = df["Date"].unique()
    df_total = df_total.sort_values(by="Date")

    for ticker in tqdm(df["Ticker"].unique()):
        df_ticker = df.loc[df.Ticker == ticker].sort_values(by="Date")
        df_ticker["Return"] = df_ticker["Adj Close"].shift(1)
        df_ticker["Return"] = np.log(df_ticker["Adj Close"] / df_ticker["Return"])

        df_total[f"Adj_Close_{ticker}"] = df_ticker[
            "Adj Close"
        ].values  # pylint: disable=unsupported-assignment-operation
        df_total[f"Return_{ticker}"] = df_ticker[
            "Return"
        ].values  # pylint: disable=unsupported-assignment-operation

    return df_total


def compute_portfolio_returns(returns, portfolio_name="Target", weights=None):
    price_columns = [i for i in returns.columns if i.startswith("Adj_Close")]
    returns[f"Adj_Close_{portfolio_name}"] = returns[price_columns].sum(axis=1) / len(
        price_columns
    )

    returns[f"Return_{portfolio_name}"] = returns[f"Adj_Close_{portfolio_name}"].shift(
        1
    )
    returns[f"Return_{portfolio_name}"] = np.log(
        returns[f"Adj_Close_{portfolio_name}"] / returns[f"Return_{portfolio_name}"]
    )

    return returns
