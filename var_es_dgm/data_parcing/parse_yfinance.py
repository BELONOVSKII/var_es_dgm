import os
import yfinance as yf
import pandas as pd


# Define a function to download data for a single stock
def download_stock_data(ticker, start_date, end_date):
    """
    Downloads historical stock data for a given ticker symbol and date range.

    Args:
        ticker (str): Ticker symbol of the stock.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        pandas.DataFrame: DataFrame containing historical stock data.
    """
    return yf.download(ticker, start=start_date, end=end_date)


def parce_data(tickers, start_date, end_date, output_path=None):
    stock_data = {}
    for ticker in tickers:
        try:
            stock_data[ticker] = download_stock_data(
                ticker, start_date, end_date
            ).reset_index()
            if output_path:
                stock_data[ticker].to_csv(
                    output_path + ticker.replace(".", "_") + ".csv", index=False
                )
            print(ticker, stock_data[ticker]["Date"].min())
        except (yf.DownloadError, KeyError):
            print(f"Error downloading data for {ticker}")


def take_trading_from_2005():
    PATH_TO_DATA = "data/stocks/"
    data_files = list(filter(lambda x: x[-4:] == ".csv", os.listdir(PATH_TO_DATA)))
    df = pd.DataFrame()
    for file_name in data_files:
        temp = pd.read_csv(PATH_TO_DATA + file_name, parse_dates=["Date"]).assign(
            Ticker=file_name[:-4]
        )

        try:
            if temp["Date"].min().year <= 2005:
                df = pd.concat([df, temp])
        except AttributeError:
            continue

    print("Number of suitable stocks: {0}".format(df["Ticker"].nunique()))
    print("\nList of availiable companies:\n", df["Ticker"].unique())
    df.to_csv("data/complete_stocks.csv", index=False)


if __name__ == "__main__":
    DATA_FOLDER = "data/"
    # Get the list of DJIA tickers (replace with your preferred source)
    tickers = (
        pd.read_excel(DATA_FOLDER + "SP-500-Companies-List.xlsx")
        .sort_values(by="Weight", ascending=False)
        .iloc[:100]  # take only 100 largest
        .loc[:, "Symbol"]
    )

    # Define start and end date (change as needed)
    start_date = "2005-01-01"
    end_date = "2023-12-31"
    parce_data(tickers, start_date, end_date, output_path=DATA_FOLDER + "stocks/")
    take_trading_from_2005()
