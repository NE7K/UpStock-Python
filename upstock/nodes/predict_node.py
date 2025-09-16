import hashlib # hash
import datetime
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from finvizfinance.news import News

from upstock.config import model_path, tokenizer_path, supabase
from upstock.models.artifacts import check_all_model, load_pickle

def run_predict():
    
    model = check_all_model(model_path, 'Sentiment Model .keras version')
    tokenizer = load_pickle(tokenizer_path, 'Tokenizer')
    
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
        print(f'finviz news parse fail : {e}')

    # string to date time
    news_df['parsed_date'] = pd.to_datetime(news_df['Date'], errors='coerce') # BUG format 지정

    today = datetime.date.today() # today
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

    predict_data = tokenizer.texts_to_sequences(predict_texts)
    predict_data = pad_sequences(predict_data, maxlen=141) # str.len result 95% 141

    prediction = model.predict(predict_data)
    # print(prediction)
    
    # past division predict data task
    # for text, prob in zip(predict_texts, prediction):
    #     label = 'positive' if prob[0] >= 0.7 else 'negative'
    #     print(f'[{label}] {text}\n : {prob[0]:.2f}\n') # :.2f
    
    #  TODO
    # 1. 중립적인 혹은, 예측이 애매한 기사 거르기 = x
    # 2. to csv 파일로 예측값과 같이 supabase에 저장
    # https://supabase.com/docs/reference/python/upsert
    
    sb_result = []  # sending to supabase predict data
        
    for text, percent in zip(predict_texts, prediction):
        # 강한 긍정과 강한 부정만 끌어다가 쓰기
        score = float(percent[0])
        if score >= 0.8:
            label = "positive"
        elif score <= 0.3:
            label = "negative"
        else:
            continue
        
        print(f"[{label}] {text}\n : {score:.2f}\n") # :.2f
    
        sb_result.append({
            'text': text,
            'percent': score, # BUG type error float32
            'label': label,
            'source': 'finviz',
            'run_at': datetime.datetime.now(datetime.timezone.utc).isoformat(), # utc time
            'hash': hashlib.sha256(text.encode("utf-8")).hexdigest() # hash 256bit -> 64자리, 16진수
        })

    if sb_result:
        try:
            response = (
                supabase.table('news_sentiment')
                .upsert(sb_result, on_conflict='hash')
                .execute()
            )
            print(f'supbase upload complete : {len(response.data)}')
            
        except Exception as e:
            print(f'supabase upload fail : {e}')
            
    else:
        print('upload data not exist')
    