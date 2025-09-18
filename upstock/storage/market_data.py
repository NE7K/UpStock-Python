import yfinance as yf
import pandas as pd

def fetch_prices(ticker:str, look_days: int = 400) -> pd.Series:
    """
    look days : 몇일치 가져올지
    interval : 일봉
    auto adjust : 병합 및 분할 고려
    """
    stock_data = yf.download(
        ticker,
        period=f'{look_days}d',
        interval='1d',
        auto_adjust=True,
        progress=True
    )
    if stock_data is None or stock_data.empty:
        return pd.Series(dtype=float)
    return stock_data['Close']