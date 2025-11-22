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
import requests
from bs4 import BeautifulSoup
import time

from tensorflow.keras.preprocessing.sequence import pad_sequences
from finvizfinance.news import News

from upstock.config import paths, supabase
from upstock.models.artifacts import load_model_safe, load_pickle

# PART market data part
from upstock.indicators.indexer import compute_index, label_zone
from upstock.storage.market_data import fetch_prices

logger = logging.getLogger(__name__)

def classify_news_tag(text: str) -> str:
    """
    뉴스 제목을 기반으로 태그를 분류합니다.
    
    Args:
        text: 뉴스 제목
        
    Returns:
        태그 문자열 (경제, 산업, 기술, 정치, 건강, 사회)
    """
    text_lower = text.lower()
    
    # 정치 키워드
    politics_keywords = [
        'politics', 'political', 'election', 'president', 'government', 'senate', 'congress',
        'democrat', 'republican', 'biden', 'trump', 'administration', 'policy', 'diplomacy',
        'international', 'trade war', 'sanctions', 'embargo', 'treaty', 'alliance', 'nato',
        'united nations', 'un', 'eu', 'brexit', 'geopolitical', 'foreign policy'
    ]
    
    # 경제 키워드
    economy_keywords = [
        'economy', 'economic', 'market', 'inflation', 'deflation', 'fed', 'federal reserve',
        'rate cut', 'rate hike', 'interest rate', 'gdp', 'unemployment', 'employment',
        'consumer price', 'cpi', 'ppi', 'retail sales', 'gross domestic product',
        'recession', 'growth', 'monetary policy', 'fiscal policy', 'treasury', 'bond',
        'stock market', 'dow', 'nasdaq', 's&p', 'sp500', 'market wrap', 'markets'
    ]
    
    # 산업 키워드
    company_keywords = [
        'company', 'companies', 'corporate', 'earnings', 'revenue', 'profit', 'loss',
        'quarterly', 'q1', 'q2', 'q3', 'q4', 'ipo', 'merger', 'acquisition', 'deal',
        'ceo', 'cfo', 'executive', 'shareholder', 'dividend', 'stock', 'shares',
        'industry', 'sector', 'automotive', 'energy', 'oil', 'gas', 'retail',
        'manufacturing', 'banking', 'finance', 'insurance'
    ]
    
    # 기술 키워드
    tech_keywords = [
        'technology', 'tech', 'ai', 'artificial intelligence', 'machine learning', 'ml',
        'innovation', 'science', 'research', 'development', 'r&d', 'semiconductor',
        'chip', 'software', 'hardware', 'cloud', 'cyber', 'digital', 'data',
        'quantum', 'blockchain', 'crypto', 'bitcoin', 'ethereum', 'nft',
        'space', 'nasa', 'rocket', 'satellite', 'electric vehicle', 'ev', 'tesla'
    ]
    
    # 사회 키워드
    social_keywords = [
        'social', 'culture', 'society', 'education', 'university', 'school',
        'climate', 'environment', 'green', 'sustainability', 'renewable',
        'immigration', 'refugee', 'human rights', 'equality', 'diversity',
        'media', 'entertainment', 'movie', 'music', 'art', 'culture'
    ]
    
    # 건강 키워드
    sports_health_keywords = [
        'sports', 'sport', 'olympics', 'nfl', 'nba', 'mlb', 'soccer', 'football',
        'basketball', 'baseball', 'tennis', 'golf', 'health', 'medical', 'medicine',
        'hospital', 'doctor', 'patient', 'disease', 'covid', 'pandemic', 'vaccine',
        'pharmaceutical', 'drug', 'treatment', 'therapy', 'fitness', 'wellness'
    ]
    
    # 우선순위에 따라 태그 분류 (경제가 가장 높은 우선순위)
    if any(keyword in text_lower for keyword in economy_keywords):
        return "경제"
    elif any(keyword in text_lower for keyword in company_keywords):
        return "산업"
    elif any(keyword in text_lower for keyword in tech_keywords):
        return "기술"
    elif any(keyword in text_lower for keyword in politics_keywords):
        return "정치"
    elif any(keyword in text_lower for keyword in sports_health_keywords):
        return "건강"
    elif any(keyword in text_lower for keyword in social_keywords):
        return "사회"
    else:
        # 기본값: 경제 (finviz 뉴스는 주로 금융/경제 관련)
        return "경제"

def fetch_finviz_news_by_type(news_type: int, source_name: str) -> pd.DataFrame:
    """
    Finviz의 특정 타입의 뉴스 페이지에서 뉴스를 가져옵니다.
    
    Args:
        news_type: 뉴스 타입 (3=Stock, 4=ETF, 5=Crypto)
        source_name: 소스 이름 (로깅용)
    
    Returns:
        DataFrame with 'Title' and 'Date' columns, or empty DataFrame if failed
    """
    try:
        url = f'https://finviz.com/news.ashx?v={news_type}'
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        news_items = []
        
        # 뉴스는 테이블 형태로 되어있음
        # nn 클래스를 가진 tr 태그에서 뉴스 추출
        for row in soup.find_all('tr', class_='nn'):
            title_elem = row.find('a')
            if title_elem:
                title = title_elem.get_text(strip=True)
                # 날짜 정보 추출 (보통 같은 행의 td에서)
                date_elem = row.find('td', class_='nn-date')
                if date_elem:
                    date_str = date_elem.get_text(strip=True)
                else:
                    # 날짜가 없으면 현재 시간 사용
                    date_str = datetime.datetime.now().strftime("%I:%M%p")
                
                news_items.append({
                    'Title': title,
                    'Date': date_str
                })
        
        if news_items:
            df = pd.DataFrame(news_items)
            logger.info(f'{source_name} News fetched: {len(df)} articles')
            return df
        else:
            logger.warning(f'No {source_name.lower()} news items found')
            return pd.DataFrame(columns=['Title', 'Date'])
            
    except Exception as e:
        logger.error(f'Failed to fetch {source_name.lower()} news: {e}')
        return pd.DataFrame(columns=['Title', 'Date'])

def fetch_stock_news() -> pd.DataFrame:
    """
    Finviz의 Stock News 페이지에서 뉴스를 가져옵니다.
    https://finviz.com/news.ashx?v=3
    
    Returns:
        DataFrame with 'Title' and 'Date' columns, or empty DataFrame if failed
    """
    return fetch_finviz_news_by_type(3, 'Stock')

def fetch_etf_news() -> pd.DataFrame:
    """
    Finviz의 ETF News 페이지에서 뉴스를 가져옵니다.
    https://finviz.com/news.ashx?v=4
    
    Returns:
        DataFrame with 'Title' and 'Date' columns, or empty DataFrame if failed
    """
    return fetch_finviz_news_by_type(4, 'ETF')

def fetch_crypto_news() -> pd.DataFrame:
    """
    Finviz의 Crypto News 페이지에서 뉴스를 가져옵니다.
    https://finviz.com/news.ashx?v=5
    
    Returns:
        DataFrame with 'Title' and 'Date' columns, or empty DataFrame if failed
    """
    return fetch_finviz_news_by_type(5, 'Crypto')

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
    
    # Market News 가져오기
    try:
        fnews = News()
        all_news = fnews.get_news()
        news_df = all_news['news'].copy()
        news_df['source'] = 'finviz_market'  # source 구분
        logger.info(f'Market News fetched: {len(news_df)} articles')
    except Exception as e:
        logger.error(f'finviz market news parse failed : {e}')
        return
    
    # 날짜 파싱 - 최근 2일치 뉴스 가져오기
    today = datetime.date.today()
    two_days_ago = today - datetime.timedelta(days=2)
    yesterday = today - datetime.timedelta(days=1)
    
    def parse_date_with_multiple_attempts(date_str):
        """여러 날짜로 시도하여 파싱"""
        # 오늘, 어제, 그제 순서로 시도
        for target_date in [today, yesterday, two_days_ago]:
            try:
                parsed = pd.to_datetime(
                    target_date.strftime("%Y-%m-%d ") + str(date_str),
                    format="%Y-%m-%d %I:%M%p",
                    errors="coerce"
                )
                if not pd.isna(parsed):
                    return parsed
            except:
                continue
        return pd.NaT  # 파싱 실패
    
    # 날짜 파싱 적용
    news_df["parsed_date"] = news_df["Date"].apply(parse_date_with_multiple_attempts)
    
    # 최근 2일치 뉴스 필터링 (날짜 파싱 실패한 뉴스도 포함 - Finviz는 최근 뉴스만 보여주므로)
    two_days_ago_datetime = pd.Timestamp(two_days_ago)  # datetime으로 변환
    recent_news = news_df[
        (news_df['parsed_date'].isna()) |  # 날짜 파싱 실패한 경우 포함
        (news_df['parsed_date'] >= two_days_ago_datetime)  # 최근 2일 이내
    ]
    
    predict_texts = recent_news['Title'].tolist()
    news_sources = recent_news['source'].tolist() if 'source' in recent_news.columns else ['finviz'] * len(predict_texts)
    
    logger.info(f'Filtered news (last 2 days + unparsed): {len(predict_texts)} articles')
    
    # past predict data
    # predict_texts = [
    #     "EM portfolios funnel near $45 billion in August but cracks are showing, IIF says", # Negative
    #     "Stocks' Bull Market Nears 3-Year Anniversary. It Likely Has More Room to Run.",    # Positive
    #     "Stock Market Today: Dow Slides As Oracle Soars; Medicare News Hits Health Leader",  # Negative
    #     "Stock Market Today: Dow and Nasdaq fall, S&P 500 loses momentum ahead of August consumer-price index on Thursday; Oracle share surge highlights technology spending", # Negative
    #     "Oracle stock booms 35%, on pace for best day since 1992", # Positive
    # ]
    
    if not predict_texts:
        logger.warning("No news found for the last 2 days")
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
        
    for text, probs, source in zip(predict_texts, prediction, news_sources):
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

        # 태그 분류
        tag = classify_news_tag(text)
        
        logger.info(f"[{label.upper()}] [{tag}] [{source}] {text} : {percent:.2f}")

        sb_result.append({
            "text": text,
            "label": label,
            "percent": percent,
            "tag": tag,
            "source": source,
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
