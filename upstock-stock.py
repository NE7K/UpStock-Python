# INFO compatibility issue import
# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping
# from keras.preprocessing.sequence import pad_sequences
# from keras.callbacks import TensorBoard

import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import os
import datetime
import supabase
import yfinance as yf

# 과거 log file 생성 규칙에 사용한 import
# import time

# validation x > split Test Data
from sklearn.model_selection import train_test_split

# supabase from
from dotenv import load_dotenv
from supabase import create_client, Client

# recent keras import
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

load_dotenv()

supabase_url = os.getenv('SupaBase_Url')
supabase_key = os.getenv('SupaBase_Key')
supabase: Client = create_client(supabase_url, supabase_key)

stock_inputdata_path = 'DataSets/upstock_nasdaq.csv'

tokenizer_path = 'SaveModel/upstock_stock_tokenizer.pickle'
model_path = 'SaveModel/upstock_stock_model.keras'
model_path_h5 = 'SaveModel/upstock_stock_model.h5'  # compatibility issue .h5

# Get Yahoo Finance Data
def load_stock(stockName, startDate):
    try:
        StockData = yf.download(
            stockName, # stock number
            start= startDate,
            auto_adjust=True, # 과거 주가와 현재 주가의 차이점을 완화 병합 혹은 분할 그리고 상승으로 인한 차이
            progress=True
        )
        
        StockData.reset_index(inplace=True)
        
        # 만약 MultiIndex 컬럼일 경우만 droplevel 수행
        if isinstance(StockData.columns, pd.MultiIndex):
            StockData.columns = StockData.columns.droplevel(1)
        
        return StockData

    except Exception as e:
        print(f'Download Fail : {e}')
        return None

# download finance data
nasdaq_df = load_stock('^NDX', '2000-01-01')
vix_df = load_stock('^VIX', '2000-01-01')

# PART rsi task
def task_RSI(data: pd.DataFrame, window: int = 14) -> pd.Series:  # 14일 기준
    delta = data['Close'].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    roll_up = pd.Series(gain).rolling(window=window).mean()
    roll_down = pd.Series(loss).rolling(window=window).mean()

    RS = roll_up / roll_down
    RSI = 100 - (100 / (1 + RS))
    return RSI

# PART macd task
def task_MACD(data: pd.DataFrame, short_window=12, long_window=26, signal_window=9):
    short_ema = data['Close'].ewm(span=short_window, adjust=False).mean()
    long_ema = data['Close'].ewm(span=long_window, adjust=False).mean()

    data['MACD'] = short_ema - long_ema
    data['Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    data['Histogram'] = data['MACD'] - data['Signal']
    return data

# indicator task
nasdaq_df['RSI'] = task_RSI(nasdaq_df, 14)
nasdaq_df = task_MACD(nasdaq_df)

# create label
nasdaq_df['Return'] = nasdaq_df['Close'].pct_change().shift(-1)
nasdaq_df['Label'] = (nasdaq_df['Return'] > 0).astype(int)

# merge nasdaq, rsi and vix
nasdaq_df['Date'] = nasdaq_df.index
vix_df['Date'] = vix_df.index
merged = pd.merge(nasdaq_df, vix_df[['Date','Close']], on='Date', how='inner', suffixes=('', '_VIX'))

# nan data delete task
merged = merged.dropna()

data = nasdaq_df.to_csv(stock_inputdata_path, index=False)

print(data)

# 과거 초기 모델에 사용했던 코드
# analyst_ratings_processed date 명시로인한 통일성 부여
# StockData.rename(columns={'Date' : 'date'}, inplace=True)












# text, chart, label 데이터 쪼개기 0.2
X_train_text, X_val_text, X_train_chart, X_val_chart, y_train, y_val = train_test_split(
    titles, chart, labels, test_size=0.2, random_state=42
)

# nomalization, 전체 데이터에서 하나의 평균과 분산을 사용
low_preprocessing = tf.keras.layers.Normalization(axis=None)
low_preprocessing.adapt(np.array(merged['Low']))
high_preprocessing = tf.keras.layers.Normalization(axis=None)
high_preprocessing.adapt(np.array(merged['High']))
open_preprocessing = tf.keras.layers.Normalization(axis=None)
open_preprocessing.adapt(np.array(merged['Open']))
close_preprocessing = tf.keras.layers.Normalization(axis=None)
close_preprocessing.adapt(np.array(merged['Close']))
volume_preprocessing = tf.keras.layers.Normalization(axis=None)
volume_preprocessing.adapt(np.array(merged['Volume']))

# nomalization result print
# normalized_close = close_preprocessing(np.array(merged['Close']))
# print(normalized_close.numpy())

# create input
low_input = tf.keras.Input(shape=(1, ), name='Low')
high_input = tf.keras.Input(shape=(1, ), name='High')
open_input = tf.keras.Input(shape=(1, ), name='Open')
close_input = tf.keras.Input(shape=(1, ), name='Close')
volume_input = tf.keras.Input(shape=(1, ), name='Volume')

x_low = low_preprocessing(low_input)
x_high = high_preprocessing(high_input)
x_open = open_preprocessing(open_input)
x_close = close_preprocessing(close_input)
x_volume = volume_preprocessing(volume_input)

# using functional api
model_input = tf.keras.Input(shape=(110,), name='model_input')
embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(model_input)
bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(embedding)
maxpool1d = tf.keras.layers.GlobalMaxPool1D()(bidirectional)
concat_layer = tf.keras.layers.Concatenate()([x_low, x_high, x_open, x_close, x_volume, maxpool1d])

dense1 = tf.keras.layers.Dense(64, activation='relu')(concat_layer)
dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
dropout2 = tf.keras.layers.Dropout(0.3)(dense2)
dense3 = tf.keras.layers.Dense(32, activation='relu')(dropout2)
model_output = tf.keras.layers.Dense(1, activation='sigmoid')(dense3)

model = tf.keras.Model(inputs=[model_input, low_input, high_input, open_input, close_input, volume_input], outputs=model_output)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_inputs = {
    'model_input' : X_train_text,
    'Low' : np.array(X_train_chart['Low']).reshape(-1, 1),
    'High' : np.array(X_train_chart['High']).reshape(-1, 1),
    'Open' : np.array(X_train_chart['Open']).reshape(-1, 1),
    'Close' : np.array(X_train_chart['Close']).reshape(-1, 1),
    'Volume' : np.array(X_train_chart['Volume']).reshape(-1, 1),
}

val_inputs = {
    'model_input': X_val_text,
    'Low': np.array(X_val_chart['Low']).reshape(-1, 1),
    'High': np.array(X_val_chart['High']).reshape(-1, 1),
    'Open': np.array(X_val_chart['Open']).reshape(-1, 1),
    'Close': np.array(X_val_chart['Close']).reshape(-1, 1),
    'Volume': np.array(X_val_chart['Volume']).reshape(-1, 1),
}

# Part callback | tensorboard --logdir=LogFile/
# time.time() 큰 숫자가 최신
#TODO 나중에 모델 딥러닝할 때 적용시킬 것 : datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") 
# tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_Model_' + str(int(time.time()))) )
tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('Stock_Model_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) )
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

# train
model.fit(train_inputs, y_train, validation_data=(val_inputs, y_val), batch_size=32, epochs=50, callbacks=[early_stop, tensorboard])
model.summary()
model.save(model_path)
# 비상용
model.save(model_path_h5)