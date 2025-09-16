"""
Config.py

dataclasses를 이용함
env 파일을 로드하고 supabase 및 각 파일의 path를 담고있음
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client
from dataclasses import dataclass   # create data class

# env load
load_dotenv()

supabase_url = os.getenv('SupaBase_Url')
supabase_key = os.getenv('SupaBase_Key')

# connect sb sdk
supabase: Client = create_client(
    supabase_url,
    supabase_key,
)

# 경로
sentiment_path = 'DataSets/upstock-sentiment-data.csv' # sentiment data
tokenizer_path = 'SaveModel/upstock_sentiment_tokenizer.pickle'
model_path = 'SaveModel/upstock_sentiment_model.keras'
model_path_h5 = 'SaveModel/upstock_sentiment_model.h5' # compatibility issue .h5

# 논문 근거
model_pkl_path = 'SaveModel/upstock_sentiment_pkl.pkl' # import matplotlib.pyplot as plt

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