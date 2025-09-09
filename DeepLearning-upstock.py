# INFO compatibility issue import
# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping
# from keras.preprocessing.sequence import pad_sequences
# from keras.callbacks import TensorBoard

import numpy as np
import pandas as pd
import pickle
import os
# 로그파일 재정립
import datetime
import yfinance as yf
# not using
import time

# validation x > split Test Data
from sklearn.model_selection import train_test_split

# recent keras import
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

price_path = 'DataSets/stock_price_data_v2.csv'
news_path = 'DataSets/analyst_ratings_processed.csv'
tokenizer_path = 'SaveModel/upstock_tokenizer.pickle'
model_path = 'SaveModel/upstock_model.keras'
# compatibility issue .h5
model_path_h5 = 'SaveModel/upstock_model.h5'

# 초기 모델에 사용한 데이터셋
# news_data = pd.read_csv('DataSets/UpStock-NewsData.csv')
# news_data = pd.read_csv('DataSets/raw_partner_headlines.csv')
# news_data2 = pd.read_csv('DataSets/raw_analyst_ratings.csv')

# 초기 모델 데이터셋 출력
# print(news_data)
# print(news_data2)
# print(news_data)

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
        
price_data = load_file(price_path, 'stock price csv file')
news_data = load_file(news_path, 'stock news csv file')

model = check_all_model(model_path, 'model .keras')
model_h5 = check_all_model(model_path_h5, 'model .h5')

tokenizer = load_pickle(tokenizer_path, 'tokenizer')

# TEST
# model.summary()

# 데이터셋 null 개수 출력
# print(news_data.isnull().sum())
# Unnamed: 0    1289
# title            0
# date             0
# stock         2578
# dtype: int64

# 길이 열 추가해서 카운트해서 lenght column에 집어넣고
# news_data['lenght'] = news_data['title'].str.len()
# print(news_data['title'].str.len().max())

# 데이터셋 길이 통계 요약 출력
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

# 초기 모델 정규화 테스트 부분
# model = tf.keras.models.Sequential([
#     # out_dim 단어를 몇 차원으로 표기할건지 중간 : 128
#     tf.keras.layers.Embeㄴdding(len(tokenizer.index_word) + 1, 64),
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
        
    # TODO tensorflow keras, keras 호환성 차이 극복용
    model = load_model("SaveModel/upstock_model.keras")
    # model.save("SaveModel/upstock_model.h5")
    
    # PART Negative : Added new text preprocessing
    # 2025-09-07 korean time 00:26 / https://finance.yahoo.com/news/bad-economic-news-might-actually-be-bad-again-100058088.html
    # 실업률 지표의 수치 상승으로 (실업률 상승) 부정적인 견해를 가진 뉴스보도
    # predict_title = ['Bad economic news might actually be bad again']
    
    # PART Positive
    # 2025-09-09 korean time 00:42 / https://finance.yahoo.com/news/live/stock-market-today-sp-500-nasdaq-dow-rise-as-wall-street-looks-ahead-to-inflation-reality-check-000205760.html
    # 25bp 인하가 아닌 50bp 인하 가능성으로 인해서 cpi 발표 전 현재 나스닥이 상승 중이라는 뉴스보도
    predict_title = ['S&P 500, Nasdaq recover from losses as markets bet on September rate cut']
    predict_title = tokenizer.texts_to_sequences(predict_title)
    predict_title = pad_sequences(predict_title, maxlen=110)
    
    # pyf_date = '2025-09-07'
    
    # # PART test stock dataset yf
    # pyf = yf.download('^NDX', start='2025-09-05', end='2025-09-09')
    # pyf.reset_index(inplace=True)
    
    # pyf['prev_Close'] = pyf['Close'].shift(1)
    # pyf['Close_logret'] = np.log(pyf['Close'] / pyf['prev_Close'])
    # pyf['Low_logret']   = np.log(pyf['Low']  / pyf['Close'])
    # pyf['High_logret']  = np.log(pyf['High'] / pyf['Close'])
    # pyf['Open_logret']  = np.log(pyf['Open'] / pyf['Close'])
    # pyf['Volume_log']   = np.log1p(pyf['Volume'])
    
    # row = pyf.loc[pyf['Date'] == pd.to_datetime()].iloc[0]

    # predict_data = {
    # 'model_input': np.array(predict_title),
    # 'Low'   : np.array([[row['Low_logret']]],   dtype=np.float32),
    # 'High'  : np.array([[row['High_logret']]],  dtype=np.float32),
    # 'Open'  : np.array([[row['Open_logret']]],  dtype=np.float32),
    # 'Close' : np.array([[row['Close_logret']]], dtype=np.float32),
    # 'Volume': np.array([[row['Volume_log']]],   dtype=np.float32),
    # }
    
    # prediction = model.predict(predict_data)
    # print(prediction, 'up' if prediction >=0.5 else 'down')
    
    pyf_date = '2025-09-07'  # 문자열 그대로 두고 필요시 바꿔 쓰기
    target_dt = pd.to_datetime(pyf_date)

        # --- PATCH START: yfinance 다운로드 ~ 예측 입력 ---
    # 1) 버퍼 넉넉하게 (전일 종가 확보), end는 exclusive
    yf_start = (target_dt - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
    yf_end   = (target_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # 2) auto_adjust=False로 일관성 유지 + 경고 해소
    pyf = yf.download('^NDX', start=yf_start, end=yf_end, auto_adjust=False, progress=False)
    pyf.reset_index(inplace=True)

    # 3) 다중컬럼 가드
    if isinstance(pyf.columns, pd.MultiIndex):
        pyf.columns = pyf.columns.droplevel(1)

    # 4) 파생 피처 계산
    pyf['prev_Close']   = pyf['Close'].shift(1)
    pyf['Close_logret'] = np.log(pyf['Close'] / pyf['prev_Close'])
    pyf['Low_logret']   = np.log(pyf['Low']   / pyf['Close'])
    pyf['High_logret']  = np.log(pyf['High']  / pyf['Close'])
    pyf['Open_logret']  = np.log(pyf['Open']  / pyf['Close'])
    pyf['Volume_log']   = np.log1p(pyf['Volume'])

    # 5) prev_Close가 NaN인 첫 행 제거 (핵심)
    pyf = pyf.dropna(subset=['prev_Close']).reset_index(drop=True)

    # 6) 타깃 날짜 = 정확히 일치 없으면 직전 거래일
    mask_eq = pyf['Date'] == target_dt
    if mask_eq.any():
        row = pyf.loc[mask_eq].iloc[0]
    else:
        prior = pyf.loc[pyf['Date'] <= target_dt]
        if len(prior) == 0:
            raise ValueError(f"{pyf_date} 기준 직전 거래일 데이터가 없습니다. 시작일을 더 과거로 늘려보세요.")
        row = prior.iloc[-1]

    # 7) 안전 체크
    req = ['Low_logret','High_logret','Open_logret','Close_logret','Volume_log']
    if pd.isna(row[req]).any():
        raise ValueError(f"필수 피처 NaN 존재(드묾): {row[req].to_dict()}")

    # 8) 예측 입력(dict)
    predict_data = {
        'model_input': np.array(predict_title),
        'Low'   : np.array([[row['Low_logret']]],   dtype=np.float32),
        'High'  : np.array([[row['High_logret']]],  dtype=np.float32),
        'Open'  : np.array([[row['Open_logret']]],  dtype=np.float32),
        'Close' : np.array([[row['Close_logret']]], dtype=np.float32),
        'Volume': np.array([[row['Volume_log']]],   dtype=np.float32),
    }

    prediction = model.predict(predict_data)
    print(f"pred_date(requested)={pyf_date}, used_date={row['Date'].date()}",
          prediction, 'up' if prediction >= 0.5 else 'down')
    
    # --- PATCH END ---

    # # yfinance end는 'exclusive'라서 +1일 해줘야 target_dt 포함됨
    # yf_start = (target_dt - pd.Timedelta(days=2)).strftime('%Y-%m-%d')  # prev_Close 계산 위해 하루 전보다 넉넉히 2일
    # yf_end   = (target_dt + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    # # 데이터 다운로드
    # pyf = yf.download('^NDX', start=yf_start, end=yf_end)
    # pyf.reset_index(inplace=True)
    # # 다중컬럼 대비(보통 ^NDX는 단일이지만 안전장치)
    # if isinstance(pyf.columns, pd.MultiIndex):
    #     pyf.columns = pyf.columns.droplevel(1)

    # # 로그 파생치 계산
    # pyf['prev_Close']   = pyf['Close'].shift(1)
    # pyf['Close_logret'] = np.log(pyf['Close'] / pyf['prev_Close'])
    # pyf['Low_logret']   = np.log(pyf['Low']   / pyf['Close'])
    # pyf['High_logret']  = np.log(pyf['High']  / pyf['Close'])
    # pyf['Open_logret']  = np.log(pyf['Open']  / pyf['Close'])
    # pyf['Volume_log']   = np.log1p(pyf['Volume'])

    # # target_dt에 정확히 일치하는 날짜가 없으면(주말/휴장) 직전 거래일로 fallback
    # mask_eq = pyf['Date'] == target_dt
    # if mask_eq.any():
    #     row = pyf.loc[mask_eq].iloc[0]
    # else:
    #     # 직전 거래일 찾기
    #     row = pyf.loc[pyf['Date'] <= target_dt].iloc[-1]

    # # 필수 피처 NaN 방어(특히 prev_Close 없는 첫행)
    # req = ['Low_logret','High_logret','Open_logret','Close_logret','Volume_log']
    # if pd.isna(row[req]).any():
    #     raise ValueError(f"필수 피처 NaN 존재: {row[req].to_dict()} (prev_Close가 없는 첫 행일 수 있음)")

    # # 예측 입력(dict)
    # predict_data = {
    #     'model_input': np.array(predict_title),
    #     'Low'   : np.array([[row['Low_logret']]],   dtype=np.float32),
    #     'High'  : np.array([[row['High_logret']]],  dtype=np.float32),
    #     'Open'  : np.array([[row['Open_logret']]],  dtype=np.float32),
    #     'Close' : np.array([[row['Close_logret']]], dtype=np.float32),
    #     'Volume': np.array([[row['Volume_log']]],   dtype=np.float32),
    # }

    # prediction = model.predict(predict_data)
    # print(f"pred_date(requested)={pyf_date}, used_date={row['Date'].date()}", prediction, 'up' if prediction >= 0.5 else 'down')
    
    # predict task
    # TEST 학습 데이터 분포 확인용
    # print(price_data['label'].value_counts(normalize=True))

else:
    print('Model file is not exists, Start DeepLearning')
    
    # Part Preprocessing
    news_data = news_data.dropna(subset=['date'])
    news_data['title'] = news_data['title'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    # utc 시간 고려
    news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce', utc=True)
    news_data['date'] = news_data['date'].dt.tz_localize(None).dt.date

    unique_text = news_data['title'].tolist()
    # 문자열 합침
    unique_text = ''.join(unique_text)
    # set 중복 제거 후 리스트 변환
    unique_text = list(set(unique_text))
    # 유니코드 순으로 나열
    unique_text.sort()

    # char level true 글자 단위, OOV 관례 : 나중에 추가되는 정규식에 없는 글자 정의
    tokenizer = Tokenizer(char_level=False, oov_token='<OOV>')

    # price data의 date 가져와서 merge
    price_data['date'] = pd.to_datetime(price_data['date']).dt.date
    merged = pd.merge(news_data, price_data[['date', 'Close_logret', 'High_logret', 'Low_logret', 'Open_logret', 'Volume_log', 'label_next_up']], on='date', how='inner')

    # merged.to_csv('DataSets/Preprocessing.csv', index=False)

    # --- 시간순 분할 + 토크나이저는 train만 ---
    merged['date'] = pd.to_datetime(merged['date'])
    merged = merged.sort_values('date').reset_index(drop=True)

    split_idx = int(len(merged) * 0.8)
    train_df = merged.iloc[:split_idx]
    val_df   = merged.iloc[split_idx:]

    # 토크나이저는 train만으로 fit
    tokenizer = Tokenizer(char_level=False, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_df['title'].tolist())
    print(len(tokenizer.index_word))

    # 텍스트 시퀀스 변환
    X_train_text = pad_sequences(
        tokenizer.texts_to_sequences(train_df['title'].tolist()), maxlen=110
    )
    X_val_text = pad_sequences(
        tokenizer.texts_to_sequences(val_df['title'].tolist()), maxlen=110
    )

    # 표 피처/라벨
    X_train_chart = train_df[['Low_logret','High_logret','Open_logret','Close_logret','Volume_log']]
    X_val_chart   = val_df[['Low_logret','High_logret','Open_logret','Close_logret','Volume_log']]

    y_train = train_df['label_next_up'].to_numpy()
    y_val   = val_df['label_next_up'].to_numpy()

    # -----------------
    # # 글자 치환
    # # 1. 타이틀을 리스트로
    # titles = merged['title'].tolist()
    # # 2. 타이틀 fit on text
    # tokenizer.fit_on_texts(titles)
    # # TODO TOKENIZER 개수
    # print(len(tokenizer.index_word))
    # # 3. 타이틀 text to sequences
    # titles = tokenizer.texts_to_sequences(titles)
    # # 4. pad sequences
    # titles = pad_sequences(titles, maxlen=110)
    # chart = merged[['Low_logret', 'High_logret', 'Open_logret', 'Close_logret', 'Volume_log']]
    # labels = np.array(merged['label_next_up'])

    # # TODO 이거 필요없으면 지우던가 EXCEPT 처리
    # # print(merged)
    # # exit()

    # # text, chart, label 데이터 쪼개기 0.2
    # X_train_text, X_val_text, X_train_chart, X_val_chart, y_train, y_val = train_test_split(
    #     titles, chart, labels, test_size=0.2, random_state=42
    # )
    # ---------------------

    # nomalization, 전체 데이터에서 하나의 평균과 분산을 사용
    # low_preprocessing = tf.keras.layers.Normalization(axis=None)
    # low_preprocessing.adapt(np.array(merged['Low_logret']))
    # high_preprocessing = tf.keras.layers.Normalization(axis=None)
    # high_preprocessing.adapt(np.array(merged['High_logret']))
    # open_preprocessing = tf.keras.layers.Normalization(axis=None)
    # open_preprocessing.adapt(np.array(merged['Open_logret']))
    # close_preprocessing = tf.keras.layers.Normalization(axis=None)
    # close_preprocessing.adapt(np.array(merged['Close_logret']))
    # volume_preprocessing = tf.keras.layers.Normalization(axis=None)
    # volume_preprocessing.adapt(np.array(merged['Volume_log']))
    
    # 7) Normalization (train만 adapt)
    low_preprocessing = tf.keras.layers.Normalization(axis=None)
    high_preprocessing = tf.keras.layers.Normalization(axis=None)
    open_preprocessing = tf.keras.layers.Normalization(axis=None)
    close_preprocessing = tf.keras.layers.Normalization(axis=None)
    volume_preprocessing = tf.keras.layers.Normalization(axis=None)

    low_preprocessing.adapt(X_train_chart['Low_logret'].to_numpy())
    high_preprocessing.adapt(X_train_chart['High_logret'].to_numpy())
    open_preprocessing.adapt(X_train_chart['Open_logret'].to_numpy())
    close_preprocessing.adapt(X_train_chart['Close_logret'].to_numpy())
    volume_preprocessing.adapt(X_train_chart['Volume_log'].to_numpy())


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
        'Low' : np.array(X_train_chart['Low_logret']).reshape(-1, 1),
        'High' : np.array(X_train_chart['High_logret']).reshape(-1, 1),
        'Open' : np.array(X_train_chart['Open_logret']).reshape(-1, 1),
        'Close' : np.array(X_train_chart['Close_logret']).reshape(-1, 1),
        'Volume' : np.array(X_train_chart['Volume_log']).reshape(-1, 1),
    }

    val_inputs = {
        'model_input': X_val_text,
        'Low': np.array(X_val_chart['Low_logret']).reshape(-1, 1),
        'High': np.array(X_val_chart['High_logret']).reshape(-1, 1),
        'Open': np.array(X_val_chart['Open_logret']).reshape(-1, 1),
        'Close': np.array(X_val_chart['Close_logret']).reshape(-1, 1),
        'Volume': np.array(X_val_chart['Volume_log']).reshape(-1, 1),
    }

    # Part callback | tensorboard --logdir=LogFile/
    # time.time() 큰 숫자가 최신
    #TODO 나중에 모델 딥러닝할 때 적용시킬 것 : datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") 
    tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_Model_' + str(int(time.time()))) )
    # tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_Model_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) )
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

    # train# 1) 디렉토리 강제 삭제
    model.fit(train_inputs, y_train, validation_data=(val_inputs, y_val), batch_size=1024, epochs=10, callbacks=[early_stop, tensorboard])
    model.summary()
    model.save(model_path)
    # 구버전용
    model.save(model_path_h5)

    try:
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
    except Exception as e:
        print('tokenizer 저장 실패')