"""
Predict node

part 1 : fetches news, runs sentiment model, uploads results to supabase
part 2 : fetches stock price, run sentiment algorithm, uploads result to supabase
"""

import hashlib # hash
import datetime
import pandas as pd
import numpy as np
import logging

from tensorflow.keras.preprocessing.sequence import pad_sequences
from finvizfinance.news import News

from upstock.config import paths, supabase
from upstock.models.artifacts import load_model_safe, load_pickle

# PART market data part
from upstock.indicators.indexer import compute_index, label_zone
from upstock.storage.market_data import fetch_prices

logger = logging.getLogger(__name__)

def run_predict():
    
    model = load_model_safe(paths.model, 'Sentiment Model (.keras) version')
    tokenizer = load_pickle(paths.tokenizer, 'Tokenizer')
    
    if model is None or tokenizer is None:
        logger.error('model or tokenizer not available')    # flow task check
        return
    
    # def load_news():
    #     fnews = News()
    #     # dictionary
    #     all_news = fnews.get_news()
    #     news_df = all_news['news']
    #     predict_texts = news_df['Title'].tolist() # predict texts
    #     # print(news_df['Title'][1])
    
    # parse except task
    try:
        fnews = News()
        all_news = fnews.get_news()
        news_df = all_news['news'] 
    except Exception as e:
        logger.error(f'finviz news parse failed : {e}')
        return

    # print(news_df['Date'].head(10))
    today = datetime.date.today() # today

    news_df["parsed_date"] = pd.to_datetime(
        today.strftime("%Y-%m-%d ") + news_df["Date"],
        format="%Y-%m-%d %I:%M%p",
        errors="coerce"
    )
    
    today_news = news_df[news_df['parsed_date'].dt.date == today] # today == parse data date
    predict_texts = today_news['Title'].tolist() # insert pare data
    
    # past predict data
    # predict_texts = [
    #     "EM portfolios funnel near $45 billion in August but cracks are showing, IIF says", # Negative
    #     "Stocks' Bull Market Nears 3-Year Anniversary. It Likely Has More Room to Run.",    # Positive
    #     "Stock Market Today: Dow Slides As Oracle Soars; Medicare News Hits Health Leader",  # Negative
    #     "Stock Market Today: Dow and Nasdaq fall, S&P 500 loses momentum ahead of August consumer-price index on Thursday; Oracle share surge highlights technology spending", # Negative
    #     "Oracle stock booms 35%, on pace for best day since 1992", # Positive
    # ]
    
    if not predict_texts:
        logger.warning("No news found for today's date")
        return

    seqs = tokenizer.texts_to_sequences(predict_texts)
    padded = pad_sequences(seqs, maxlen=300, padding='post', truncating='post')
    prediction = model.predict(padded)
    # print(prediction)
    
    # past division predict data task
    # for text, prob in zip(predict_texts, prediction):
    #     label = 'positive' if prob[0] >= 0.7 else 'negative'
    #     print(f'[{label}] {text}\n : {prob[0]:.2f}\n') # :.2f
    
    #  TODO
    # 1. 중립적인 혹은, 예측이 애매한 기사 거르기 = x
    # 2. to csv 파일로 예측값과 같이 supabase에 저장
    # https://supabase.com/docs/reference/python/upsert
    
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    sb_result = [] # sending to supabase predict data

    pos, neg, neu = 0, 0, 0
        
    for text, probs in zip(predict_texts, prediction):
        pred_idx = int(np.argmax(probs))
        label = label_map[pred_idx]
        percent = float(np.max(probs))  # 가장 높은 softmax 확률

        # 감정별 카운트
        if label == "positive":
            pos += 1
        elif label == "negative":
            neg += 1
        else:
            neu += 1

        logger.info(f"[{label.upper()}] {text} : {percent:.2f}")

        sb_result.append({
            "text": text,
            "label": label,
            "percent": percent,
            "source": "finviz",
            "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })

    if sb_result:
        try:
            response = (
                supabase.table('news_sentiment')
                .upsert(sb_result, on_conflict='hash')
                .execute()
            )
            logger.info(f'supbase upload complete : {len(response.data)} rows')
            
        except Exception as e:
            logger.error(f'supabase upload failed : {e}')
            
    else:
        logger.warning('upload data not exist')
    
    # market sentiment part
    try:
        spy = fetch_prices("SPY")
        vix = fetch_prices("^VIX")

        index_val, comp = compute_index(pos, neg, spy, vix)
        zone = label_zone(index_val)

        logger.info(
            f"Market Index : {index_val} ({zone}) "
            f"[pos={pos}, neu={neu}, neg={neg}] "
            f"[rsi {comp['s_rsi']:+.2f}, vix {comp['s_vix']:+.2f}, "
            f"macd {comp['s_macd']:+.2f}, ma {comp['s_ma']:+.2f}]"
        )

        recode = {
            "date_utc": datetime.datetime.now(datetime.timezone.utc).date().isoformat(),
            "score": int(index_val),
            "zone": zone,
            "rsi": comp.get("rsi", 0.0),
            "s_rsi": comp.get("s_rsi", 0.0),
            "vix": comp.get("vix", 0.0),
            "s_vix": comp.get("s_vix", 0.0),
            "macd_val": comp.get("macd_val", 0.0),
            "signal_val": comp.get("signal_val", 0.0),
            "s_macd": comp.get("s_macd", 0.0),
            "s_ma": comp.get("s_ma", 0.0),
            "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        }

        supabase.table("market_sentiment_index").upsert(
            recode, on_conflict="date_utc"
        ).execute()

        logger.info("Supabase market sentiment upload complete")

    except Exception as e:
        logger.error(f"Market index update failed: {e}")
