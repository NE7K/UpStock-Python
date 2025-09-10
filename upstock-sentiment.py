# INFO compatibility issue import
# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping
# from keras.preprocessing.sequence import pad_sequences
# from keras.callbacks import TensorBoard

import numpy as np
import pandas as pd
import pickle
import os

# 과거 log file 생성 규칙에 사용한 import
import time
import datetime # 보기 편한 log file 생성 시 필요한 import

# validation x > split Test Data
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# 경로
sentiment_path = 'DataSets/upstock-sentiment-data.csv' # sentiment data
tokenizer_path = 'SaveModel/upstock_sentiment_tokenizer.pickle'
model_path = 'SaveModel/upstock_sentiment_model.keras'
model_path_h5 = 'SaveModel/upstock_sentiment_model.h5' # compatibility issue .h5

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

# INFO load tokenizer => TextVectorization로 변경 가능성 유의
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

# INFO 초기 모델 정규화 테스트 부분
# model = tf.keras.models.Sequential([
#     # out_dim 단어를 몇 차원으로 표기할건지 중간 : 128
#     tf.keras.layers.Embedding(len(tokenizer.index_word) + 1, 64),
#     # return sequence 다음 레이어에서 모든 sequence 사용 가능하게
#     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
#     # 앞선 레이어에서 중요한 단어 (최대값 이용)
#     tf.keras.layers.GlobalAveragePooling1D(),
#     tf.keras.layers.Dense(64, activation='relu'),
#     # over fitting 방지
#     tf.keras.layers.Dropout(0.3),
#     tf.keras.layers.Dense(1, activation='sigmoid')
# ])

# save model exists => predict
# save model not exists => DeepLearning
if os.path.exists(model_path) and os.path.exists(tokenizer_path):

    # TEST predict data set
    predict_data = {
        "EM portfolios funnel near $45 billion in August but cracks are showing, IIF says", # Negative
        "Stocks' Bull Market Nears 3-Year Anniversary. It Likely Has More Room to Run.",    # Positive
        "Stock Market Today: Dow Slides As Oracle Soars; Medicare News Hits Health Leader",  # Negative
        "Stock Market Today: Dow and Nasdaq fall, S&P 500 loses momentum ahead of August consumer-price index on Thursday; Oracle share surge highlights technology spending", #부정
        "Oracle stock booms 35%, on pace for best day since 1992" # 긍정
    }

    predict_data = tokenizer.texts_to_sequences(predict_data)
    predict_data = pad_sequences(predict_data, maxlen=110) # str.len result 75% : 106

    prediction = model.predict(predict_data)
    print(prediction)
    
    # TODO
    # for i, sentence in enumerate(test_sentences):
    # prob = predictions[i][0]
    # label = "긍정 (1)" if prob >= 0.5 else "부정 (0)"
    # print(f"{sentence} -> {prob:.4f} ({label})")

else:
    print('Sentiment Model and Tokenizer is not exists, Start DeepLearning')
    
    # TEST sentiment data regex : 필요없으면 빼보는 것도 괜찮음
    # sentiment_data['Text'] = sentiment_data['Text'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    # sentiment_data['Text'] = sentiment_data['Text'].str.replace('[^a-zA-Z0-9$%+\- ]', '', regex=True)

    # PART word level
    sentiment_text = sentiment_data['Text'].astype(str).tolist()
    label = sentiment_data['Sentiment'].values

    # Tokenizer
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentiment_text)
    sequences = tokenizer.texts_to_sequences(sentiment_text)
    sentiment_text = pad_sequences(sequences, maxlen=110)

    # text, sentiment data split 0.2
    X_train, X_val, y_train, y_val = train_test_split(sentiment_text, label, test_size=0.2, random_state=42)

    # using functional api
    model_input = tf.keras.Input(shape=(110,), name='model_input')
    embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(model_input)
    bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(embedding)
    maxpool1d = tf.keras.layers.GlobalMaxPool1D()(bidirectional)
    dense1 = tf.keras.layers.Dense(64, activation='relu')(maxpool1d)
    dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
    dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout1) # 추가 데이터 확보시 Added dense layer add
    model_output = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)
    
    model = tf.keras.Model(inputs=model_input, outputs=model_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # INFO callback | tensorboard --logdir=LogFile/
    # time.time() 큰 숫자가 최신
    # TODO 가독성 좋지 못하면 아래로 대체
    # tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_SentimentModel_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_SentimentModel_' + str(int(time.time()))) )
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1) # early stop alarm

    # 학습
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=64,
        epochs=20,
        callbacks=[early_stop, tensorboard]
    )

    # TEST
    model.summary()
    
    # save
    try:
        model.save(model_path)
        model.save(model_path_h5)
    except Exception as e:
        print(f'model save fail : {e}')

    try:
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
    except Exception as e:
            print(f'tokenizer save fail : {e}')
