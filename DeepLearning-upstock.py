import tensorflow as tf
import pandas as pd
from keras.preprocessing.text import Tokenizer
import pickle
import numpy as np

try:
    price_data = pd.read_csv('DataSets/stock_price_data.csv')
except Exception as e:
    print('주식 가격 csv 파일 불러오기 실패')

# 초기 모델에 사용한 데이터셋
# news_data = pd.read_csv('DataSets/UpStock-NewsData.csv')

# news_data = pd.read_csv('DataSets/raw_partner_headlines.csv')
# news_data2 = pd.read_csv('DataSets/raw_analyst_ratings.csv')
try:
    news_data = pd.read_csv('DataSets/analyst_ratings_processed.csv')
except Exception as e:
    print('뉴스 기사 불러오기 실패')

# print(news_data)
# print(news_data2)
# print(news_data)

news_data = news_data.dropna(subset=['date'])

# print(news_data.isnull().sum())

# Unnamed: 0    1289
# title            0
# date             0
# stock         2578
# dtype: int64

# title, date : 전처리할때 date stamp 변경, "나' 제거해야함(특수문자)
# x = news_data
# date, label 
# y = price_data

news_data['title'] = news_data['title'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
# utc 시간 고려
news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce', utc=True)
news_data['date'] = news_data['date'].dt.tz_localize(None).dt.date

# print(news_data['date'])

# print(price_data['date'])

# dd = news_data['title'].duplicated().sum()
# print(dd)

unique_text = news_data['title'].tolist()
# 문자열 합침
unique_text = ''.join(unique_text)
# set 중복 제거 후 리스트 변환
unique_text = list(set(unique_text))
# 유니코드 순으로 나열
unique_text.sort()

# print(unique_text)

# char level true 글자 단위, OOV 관례 : 나중에 추가되는 정규식에 없는 글자 정의
tokenizer = Tokenizer(char_level=True, oov_token='<OOV>')

# 글자 치환
context_list = news_data['title'].tolist()
tokenizer.fit_on_texts(context_list)
print(tokenizer.index_word)

# 맵핑한 숫자에 맞게 치환
news_list = news_data['title'].to_list()
train_x = tokenizer.texts_to_sequences(news_list)

with open('SaveModel/upstock_tokenizer.pickle', 'wb') as f:
    pickle.dump(tokenizer, f)

# print(price_data['label'].tolist())
train_y = np.array(price_data['label'].tolist())
# print(trian_y)