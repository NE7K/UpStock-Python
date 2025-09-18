"""
Indicators core part
"""

import numpy as np
import pandas as pd

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    rsi : RSI = 100 - (100 / (1 + RS))
    .diff() : 연속된 값의 차이
    lower=0 : 0보다 작은 값 전부 0
    rolling : 묶음
    """
    delta = series.diff()   # 하루 전 대비 변화량
    gain = (delta.clip(lower=0)).rolling(period).mean() # 평균 14일 상승분
    loss = (-delta.clip(upper=0)).rolling(period).mean() # 평균 14일 하락분
    rs = gain / (loss.replace(0, np.nan))   # 평균 상승폭 / 평균 하락폭, 0 -> nan(결측치)
    out = 100 - (100 / (1 + rs))    # RSI = 100 - (100 / (1 + RS))
    return out

def score_rsi(rsi_value: float) -> float:
    # Rsi는 70이상은 과매수 30이하는 과매도이기 때문에 30=+1, 50=0, 70=-1
    return float(np.clip((50.0 - rsi_value) / 20.0, -1.0, 1.0 ))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    """
    단기 지수이동평균에서 장기 EMA를 빼서 구한 차이값
    단기 = fast = 12
    장기 = slow = 26
    차이값(DIF) = signal = 9
    """
    exp1 = series.ewm(span=fast, adjust=False).mean()   # ewm 메서드 지수 가중 이동평균으로 단기 EMA
    exp2 = series.ewm(span=slow, adjust=False).mean()   # 장기 EMA
    macd_line = exp1 - exp2 # macd line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()   # signal line
    hist = macd_line - signal_line  # 히스토그램
    return macd_line, signal_line, hist

def score_macd(macd_line: float, signal_line: float, hist_series: pd.Series) -> float:
    """
    macd - signal 히스토그램 -1 ~ +1 return
    """
    hist = macd_line - signal_line
    
    if hist_series.empty:
        return 0.0  # float

    min_val = float(hist_series.min().iloc[0] if hasattr(hist_series.min(), "iloc") else hist_series.min())  # 과거 히스토그램 값 중 최소
    max_val = float(hist_series.max().iloc[0] if hasattr(hist_series.max(), "iloc") else hist_series.max())  # 가장 큰 양수

    scale = max(abs(min_val), abs(max_val), 1e-9)   # abs 절댓값 1e-9로 0으로 나누는 것 방지
    hist_val = hist.iloc[-1] if hasattr(hist, "iloc") else hist
    return float(np.clip(hist_val / scale, -1, 1))
    # return float(np.clip(hist / scale, -1, 1))

def score_vix(vix_today: float, hist_vix: pd.Series) -> float:
    """
    VIX 낮으면 변동 가능성 낮음
    p25 : 25%가 이 값 이하
    p75 : 75%가 이 값 이하
    """
    vals = hist_vix.dropna()    # hist vix nan 제거
    if vals.empty:
        return 0.0
    
    p25, p75 = np.nanpercentile(vals, [25, 75]) # 100등분
    if vix_today <= p25:
        return 1.0
    if vix_today >= p75:
        return -1.0
    return 1.0 - 2.0 * ((vix_today - p25) / max(1e-9, (p75 - p25))) # 중간 구간 점수화

def score_news(pos: int, neg: int) -> float:
    """
    긍정 혹은 부정 개수 파악 후 리턴
    """
    tot = max(1, pos + neg)
    return (pos - neg) / tot

def score_moving_average(price_series: pd.Series, window: int = 200) -> float:
    """
    이동평균성 200일 기준
    종가의 위치에 따른 점수
    """
    if price_series.empty:
        return 0.0
    
    last_price = price_series.iloc[-1]  # 최근 종가 iloc 마지막 값
    price = last_price.item() if hasattr(last_price, 'item') else float(last_price)
    
    ma_series = price_series.rolling(window=window).mean()
    if ma_series.dropna().empty:
        return 0.0
    
    last_ma = ma_series.iloc[-1]
    ma = last_ma.item() if hasattr(last_ma, 'item') else float(last_ma)
    
    diff = (price - ma) / ma
    return float(np.clip(diff / 0.1, -2, 2))    # +-10이면 +-1점 더 크면 +-2까지 허용

