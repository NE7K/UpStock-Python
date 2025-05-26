import tensorflow as tf
import pandas as pd

price_data = pd.read_csv('DataSets/stock_price_data.csv')

# 초기 모델에 사용한 데이터셋
# news_data = pd.read_csv('DataSets/UpStock-NewsData.csv')

# news_data = pd.read_csv('DataSets/raw_partner_headlines.csv')
# news_data2 = pd.read_csv('DataSets/raw_analyst_ratings.csv')
news_data = pd.read_csv('DataSets/analyst_ratings_processed.csv')

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

news_data['title'] = news_data['title'].str.replace('[^a-zA-z ]', '', regex=True)

news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce', utc=True)
news_data['date'] = news_data['date'].dt.tz_localize(None).dt.date

# print(news_data['date'])

# print(price_data['date'])

dd = news_data['title'].duplicated().sum()

print(dd)