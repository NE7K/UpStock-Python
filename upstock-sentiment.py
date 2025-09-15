# INFO compatibility issue import
# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping
# from keras.preprocessing.sequence import pad_sequences
# from keras.callbacks import TensorBoard

# pip github connect | pip freeze > piplist.txt
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
import time
import datetime

import supabase
import hashlib # hash
from supabase import create_client, Client
from dotenv import load_dotenv # env with os

# validation x > split Test Data
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from finvizfinance.news import News

# env load
load_dotenv()

supabase_url = os.getenv('SupaBase_Url')
supabase_key = os.getenv('SupaBase_Key')

# connect sb sdk
supabase: Client = create_client(supabase_url, supabase_key)

# supabase storage
def download_model_file():
    bucket_name = 'sentiment_file'
    
    file_paths = [
        'upstock_sentiment_model.keras',
        'upstock_sentiment_model.h5',
        'upstock_sentiment_tokenizer.pickle'
    ]

    os.makedirs('SaveModel', exist_ok=True) # exist no error
    bucket = supabase.storage.from_(bucket_name)
    
    for file_path in file_paths:
        try:
            res = bucket.download(file_path)

            # download() result bytes > res 사용, result response > res.read()실행
            content = res.read() if hasattr(res, "read") else res
            local_path = os.path.join("SaveModel", os.path.basename(file_path))
            
            with open(local_path, "wb") as f:
                f.write(content)

            print(f"{file_path} download complete {local_path}")

        except Exception as e:
            print(f"{file_path} download fail : {e}")

# download_model_file()

# 경로
sentiment_path = 'DataSets/upstock-sentiment-data.csv' # sentiment data
tokenizer_path = 'SaveModel/upstock_sentiment_tokenizer.pickle'
model_path = 'SaveModel/upstock_sentiment_model.keras'
model_path_h5 = 'SaveModel/upstock_sentiment_model.h5' # compatibility issue .h5

# TODO 근거
model_pkl_path = 'SaveModel/upstock_sentiment_pkl.pkl' # import matplotlib.pyplot as plt

# load file < predict task에는 필요없음
def load_file(path, description):
    if os.path.exists(path):
        try:
            print(f'{description} load complete')
            return pd.read_csv(path)
        except Exception as e:
            print(f'{description} exists but, load fail {e}')
            return None
    else:
        print(f'{description} not exists')
        return None

# load save model : https://www.tensorflow.org/guide/keras/save_and_serialize?hl=ko#savedmodel_%ED%98%95%EC%8B%9D
# model load 지침
def check_all_model(path, description):
    if os.path.exists(path):
        try:
            model = load_model(path)
            print(f'{description} load complete')
            return model
        except Exception as e:
            print(f'{description} load fail : {e}')
            return None
    else:
        print(f'{description} not exists')
        return None

# load tokenizer => TextVectorization로 변경 가능성 유의
def load_pickle(path, description):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                tokenizer = pickle.load(f)
                print(f'{description} load complete')
                return tokenizer
        except Exception as e:
            print(f'{description} load fail : {e}')
            return None
    else:
        print(f'{description} not exists')
        return None

# dataset
sentiment_data = load_file(sentiment_path, 'Sentiment csv file')

# model
model = check_all_model(model_path, 'Sentiment Model .keras version')
model_h5 = check_all_model(model_path_h5, 'Sentiment Model .h5 version')

tokenizer = load_pickle(tokenizer_path, 'Tokenizer')

# TEST
# model.summary()

# 데이터셋 null 개수 출력
# print(news_data.isnull().sum())
# RESULT
# Text         0
# Sentiment    0
# dtype: int64

# 길이 열 추가해서 카운트하고 싶으면 Added lenght column
# sentiment_data['lenght'] = sentiment_data['Text'].str.len()
# print(sentiment_data['Text'].str.len().max())
# RESULT 154

# 데이터셋 길이 통계 요약 출력
# print(sentiment_data['Text'].str.len().describe())
# RESULT
# count    5791.000000
# mean       78.507857
# std        37.409135
# min         6.000000
# 25%        48.000000
# 50%        79.000000
# 75%       106.000000
# max       154.000000
# Name: Text, dtype: float64

# RESULT 데이터셋 maxlen 95
# lengths = sentiment_data['Text'].str.len()
# print(lengths.quantile(0.90)) # 133.0
# print(lengths.quantile(0.95)) # 141.0
# exit()

# save model exists => predict
# save model not exists => DeepLearning
if os.path.exists(model_path) and os.path.exists(tokenizer_path):
    
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
    
    
else:
    print('Sentiment Model and Tokenizer is not exists, Start DeepLearning')
    
    # TEST sentiment data regex
    # sentiment_data['Text'] = sentiment_data['Text'].str.replace('[^a-zA-Z0-9$%+\- ]', '', regex=True)

    # PART word level
    sentiment_text = sentiment_data['Text'].astype(str).tolist()
    label = sentiment_data['Sentiment'].values

    # Tokenizer
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentiment_text)
    sequences = tokenizer.texts_to_sequences(sentiment_text)
    sentiment_text = pad_sequences(sequences, maxlen=141)

    # text, sentiment data split 0.2
    X_train, X_val, y_train, y_val = train_test_split(sentiment_text, label, test_size=0.2, random_state=42)

    # using functional api
    model_input = tf.keras.Input(shape=(141,), name='model_input')
    embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(model_input)
    bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(embedding)
    maxpool1d = tf.keras.layers.GlobalMaxPool1D()(bidirectional)
    dense1 = tf.keras.layers.Dense(64, activation='relu')(maxpool1d)
    dropout1 = tf.keras.layers.Dropout(0.3)(dense1) # overfitting 
    dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout1) # 추가 데이터 확보시 Added dense layer add
    model_output = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)
    
    model = tf.keras.Model(inputs=model_input, outputs=model_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # INFO callback | tensorboard --logdir=LogFile/
    # time.time() 큰 숫자가 최신
    # TODO 가독성 좋지 못하면 아래로 대체
    # tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_SentimentModel_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_SentimentModel_' + str(int(time.time()))) )
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1) # early stop alarm

    # history => pkl 
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=20,
        callbacks=[early_stop, tensorboard]
    )

    model.summary()
    
    # save
    try:
        model.save(model_path)
        # model.save(model_path_h5) # h5 version
    except Exception as e:
        print(f'model save fail : {e}')

    try:
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
    except Exception as e:
            print(f'tokenizer save fail : {e}')

    try:
        with open(model_pkl_path, 'wb') as f:
            pickle.dump(history.history, f) # matplot
    except Exception as e:
            print(f'model pkl save fail : {e}')
            
            
