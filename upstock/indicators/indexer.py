import numpy as np
import pandas as pd
from upstock.indicators.core import (
    rsi, macd,
    score_rsi, score_macd, score_vix, score_news, score_moving_average  # 점수화
)

def compute_index(
  news_pos: int, news_neg: int,
  price_series: pd.Series,
  vix_series: pd.Series,
  w_news=0.35, w_rsi=0.15, w_vix=0.15, w_macd=0.15, w_ma=0.20,  
):
    """
    가중치
    뉴스 0.35, rsi 0.15, vix 0.15, macd 0.15, 이동평균선 0.20
    """
    
    S_news = score_news(news_pos, news_neg) # news part
    
    rsi_series = rsi(price_series, 14)  # rsi 14일 계산
    if not rsi_series.dropna().empty:
        rsi_val = rsi_series.dropna().iloc[-1].item()
        S_rsi = score_rsi(rsi_val)
    else:
        rsi_val, S_rsi = None, 0.0
    
    if not vix_series.dropna().empty:
        vix_val = vix_series.dropna().iloc[-1].item()
        S_vix = score_vix(vix_val, vix_series.dropna())
    else:
        vix_val, S_vix = None, 0.0
        
    macd_line, signal_line, hist_series = macd(price_series)    # macd
    macd_val, signal_val = macd_line.iloc[-1], signal_line.iloc[-1].item()
    S_macd = score_macd(macd_val, signal_val, hist_series.dropna())
    
    S_ma = score_moving_average(price_series, window=200)   # 이동평균선
    
    weights, signals = [], []   # 가중치, 점수
    if news_pos + news_neg > 0: # new 1개 이상 존재시 반영
        weights.append(w_news); signals.append(S_news)
    if rsi_val is not None: # rsi 존재시 반영
        weights.append(w_rsi); signals.append(S_rsi)
    if vix_val is not None: # vix 존재시 반영
        weights.append(w_vix); signals.append(S_vix)
    weights.append(w_macd); signals.append(S_macd)  # macd 반영
    weights.append(w_ma); signals.append(S_ma)  # 이동평균선 반영
    
    W = sum(weights) or 1.0 # 가중치 합 1.0은 리스트 비어있으면 혹은 합계 0이면 분모 0되는거 방지
    S = sum(s * ( w / W) for s, w in zip(signals, weights))
    index_raw = round((S + 2.0) * 25.0) # 0~100 범위로 매핑
    
    comp = {
        'news_pos': news_pos,
        'news_neg': news_neg,
        's_news': S_news,
        'rsi_val': rsi_val,
        's_rsi': S_rsi,
        'vix_val': vix_val,
        's_vix': S_vix,
        'macd_val': float(macd_val),
        'signal_val': float(signal_val),
        's_macd': S_macd,
        's_ma': S_ma
    }
    return index_raw, comp

def label_zone(idx: int) -> str:
    if idx <= 20: return "Distressed"
    if idx <= 40: return "Degensive"
    if idx <= 59: return "Neutral"
    if idx <= 80: return "Speculative"
    return "Overheated"