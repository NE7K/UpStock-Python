import tensorflow as tf
import pandas as pd
import pickle
import numpy as np
import time

# validation x > test data
from sklearn.model_selection import train_test_split

# pc import
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.callbacks import TensorBoard
# mac import
# from keras.preprocessing.text import Tokenizer

# from keras.callbacks import EarlyStopping

# from keras.preprocessing.sequence import pad_sequences

# from keras.callbacks import TensorBoard

# if 파일이 존재하면, 

price_path = 'DataSets/stock_price_data.csv'
news_path = 'DataSets/analyst_ratings_processed.csv'
tokenizer_path = 'SaveModel/upstock_tokenizer.pickle'
model_path = 'SaveModel/upstock_model.keras'

try:
    price_data = pd.read_csv(price_path)
except Exception as e:
    print('주식 가격 csv 파일 불러오기 실패')

# 초기 모델에 사용한 데이터셋
# news_data = pd.read_csv('DataSets/UpStock-NewsData.csv')

# news_data = pd.read_csv('DataSets/raw_partner_headlines.csv')
# news_data2 = pd.read_csv('DataSets/raw_analyst_ratings.csv')
try:
    news_data = pd.read_csv(news_path)
except Exception as e:
    print('뉴스 기사 불러오기 실패')

# print(news_data)
# print(news_data2)
# print(news_data)

# 중복 제거거
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
tokenizer = Tokenizer(char_level=False, oov_token='<OOV>')

# price data의 date 가져와서 merge
price_data['date'] = pd.to_datetime(price_data['date']).dt.date
merged = pd.merge(news_data, price_data[['date', 'label']], on='date', how='inner')

# 글자 치환
news_context_list = merged['title'].tolist()
tokenizer.fit_on_texts(news_context_list)
# print(len(tokenizer.index_word))

train_x = tokenizer.texts_to_sequences(news_context_list)

# 길이 열 추가해서 카운트해서 lenght column에 집어넣고
news_data['lenght'] = news_data['title'].str.len()
# print(news_data['title'].str.len().max())

# print(news_data['title'].str.len().describe())

# count    1.399180e+06
# mean     6.991809e+01
# std      3.899507e+01
# min      1.000000e+00
# 25%      4.500000e+01
# 50%      6.100000e+01
# 75%      8.200000e+01
# max      5.000000e+02
# Name: title, dtype: float64

train_x = pad_sequences(train_x, maxlen=110)
train_y = np.array(merged['label'])
# print(trian_y)

try:
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
except Exception as e:
    print('tokenizer 저장 실패')
    
trainx, valx, trainy, valy = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    # out_dim 단어를 몇 차원으로 표기할건지 중간 : 128
    tf.keras.layers.Embedding(len(tokenizer.index_word) + 1, 64),
    
    # return sequence 다음 레이어에서 모든 sequence 사용 가능하게
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    
    # 앞선 레이어에서 중요한 단어 (최대값 이용)
    tf.keras.layers.GlobalAveragePooling1D(),
    
    tf.keras.layers.Dense(64, activation='relu'),
    
    # over fitting 방지
    tf.keras.layers.Dropout(0.3),
    
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# save log
tensorboard = tf.keras.callbacks.TensorBoard(log_dir='LogFile/Log{}'.format('_Model_' + str(int(time.time()))) )

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

model.fit(trainx, trainy, validation_data=(valx, valy), batch_size=64, epochs=1, callbacks=[early_stop, tensorboard])

model.save(model_path)

model.summary()

# tensorboard --logdir=logs/fit