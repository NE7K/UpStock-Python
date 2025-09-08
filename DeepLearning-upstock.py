# INFO compatibility issue import
# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping
# from keras.preprocessing.sequence import pad_sequences
# from keras.callbacks import TensorBoard

import numpy as np
import pandas as pd
import pickle
import time
import os
# 로그파일 재정립
import datetime

# validation x > split Test Data
from sklearn.model_selection import train_test_split

# recent keras import
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

price_path = 'DataSets/stock_price_data.csv'
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
        
        
    # TODO 전처리 필요, 2025 절대가격 그대로 들어가지 않도록 
    
    
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
    
    # PART Negative : Added new stock price data
    # Price              Close      High           Low          Open      Volume                                             
    # 2025-09-05  23652.439453  23860.25  23475.330078  23841.980469  8413730000
    # predict_data = {
    #     'model_input': np.array(predict_title),  # (1, 110)
    #     'Low': np.array([[23475.330078]], dtype=np.float32),
    #     'High': np.array([[23860.25]], dtype=np.float32),
    #     'Open': np.array([[23841.980469]], dtype=np.float32),
    #     'Close': np.array([[23652.439453]], dtype=np.float32),
    #     'Volume': np.array([[8413730000]], dtype=np.float32),
    # }
    
    # PART Positive : Added new stock price data 
    # Price              Close          High           Low          Open     Volume
    # Ticker              ^NDX          ^NDX          ^NDX          ^NDX       ^NDX
    # Date                                                                         
    # 2025-09-08  23825.947266  23851.808594  23737.943359  23759.599609  474357663

    predict_data = {
        'model_input': np.array(predict_title),  # (1, 110)
        'Low': np.array([[23737.943359]], dtype=np.float32),
        'High': np.array([[23851.808594]], dtype=np.float32),
        'Open': np.array([[23759.599609]], dtype=np.float32),
        'Close': np.array([[23825.947266]], dtype=np.float32),
        'Volume': np.array([[474357663]], dtype=np.float32),
    }
    
    prediction = model.predict(predict_data)
    print(prediction, 'up' if prediction >=0.5 else 'down')
    
    #PART predict task
    # 학습 데이터 분포 확인용
    print(price_data['label'].value_counts(normalize=True))

else:
    print('Model file is not exists, Start DeepLearning (v2 with returns-features & next-day label)')

    # ---------- 1) 뉴스 전처리 ----------
    news_data = news_data.dropna(subset=['date'])
    news_data['title'] = news_data['title'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce', utc=True)
    news_data['date'] = news_data['date'].dt.tz_localize(None).dt.date

    # ---------- 2) 가격 라벨/특징 생성 (Next-Day label + 비율 특성) ----------
    price_data['date'] = pd.to_datetime(price_data['date']).dt.date

    # next-day label: 내일 종가가 오늘 종가보다 높으면 1(상승), 아니면 0(하락)
    price_data['label'] = (price_data['Close'].shift(-1) > price_data['Close']).astype(int)

    # 비율/수익률 기반 특징 (스케일 불변)
    eps = 1e-12
    price_data['ret']       = (price_data['Close'] - price_data['Open']) / (price_data['Open'] + eps)
    price_data['range']     = (price_data['High']  - price_data['Low'])  / (price_data['Open'] + eps)
    price_data['up_tail']   = (price_data['High']  - price_data['Close'])/ (price_data['Open'] + eps)
    price_data['down_tail'] = (price_data['Open']  - price_data['Low'])  / (price_data['Open'] + eps)
    price_data['log_vol']   = np.log1p(price_data['Volume'].clip(lower=0))

    # 라벨/특징 생성 후 마지막 행(shift로 NaN 생김) 제거
    price_feats = price_data[['date','ret','range','up_tail','down_tail','log_vol','label']].dropna()

    # ---------- 3) 날짜 기준 머지 ----------
    merged = pd.merge(
        news_data[['date','title']],
        price_feats,
        on='date',
        how='inner'
    ).sort_values('date')

    # ---------- 4) 시계열 분할 (예: Train<=2022, Val=2023, Test>=2024) ----------
    merged['date'] = pd.to_datetime(merged['date'])
    train_end = pd.to_datetime('2022-12-31')
    val_end   = pd.to_datetime('2023-12-31')

    train_df = merged[merged['date'] <= train_end]
    val_df   = merged[(merged['date'] > train_end) & (merged['date'] <= val_end)]
    test_df  = merged[merged['date'] > val_end]

    # ---------- 5) 텍스트 토크나이즈 (기존 Tokenizer 유지) ----------
    tokenizer = Tokenizer(char_level=False, oov_token='<OOV>')
    tokenizer.fit_on_texts(train_df['title'].tolist())

    def texts_to_padded(titles, maxlen=110):
        seqs = tokenizer.texts_to_sequences(titles.tolist())
        return pad_sequences(seqs, maxlen=maxlen)

    X_train_text = texts_to_padded(train_df['title'])
    X_val_text   = texts_to_padded(val_df['title'])
    X_test_text  = texts_to_padded(test_df['title'])

    # ---------- 6) 수치 특징/라벨 ----------
    num_cols = ['ret','range','up_tail','down_tail','log_vol']
    X_train_num = train_df[num_cols].to_numpy(dtype=np.float32)
    X_val_num   = val_df[num_cols].to_numpy(dtype=np.float32)
    X_test_num  = test_df[num_cols].to_numpy(dtype=np.float32)

    y_train = train_df['label'].to_numpy(dtype=np.float32)
    y_val   = val_df['label'].to_numpy(dtype=np.float32)
    y_test  = test_df['label'].to_numpy(dtype=np.float32)

    # ---------- 7) 정규화 레이어 (훈련셋으로만 adapt) ----------
    # 5개의 단일 입력으로 나누던 방식 → 이번엔 하나의 벡터 입력(N=5)로 단순화
    num_input = tf.keras.Input(shape=(5,), name='num_input')
    num_norm  = tf.keras.layers.Normalization(axis=-1, name='num_norm')
    num_norm.adapt(X_train_num)

    x_num = num_norm(num_input)

    # ---------- 8) 텍스트 입력/임베딩 ----------
    text_input = tf.keras.Input(shape=(110,), name='text_input')
    vocab_size = len(tokenizer.word_index) + 1
    x_text = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, name='embedding')(text_input)
    x_text = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True), name='bilstm')(x_text)
    x_text = tf.keras.layers.GlobalMaxPool1D(name='maxpool')(x_text)

    # ---------- 9) 결합 & 분류기 ----------
    x = tf.keras.layers.Concatenate(name='concat')([x_num, x_text])
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='out')(x)

    model = tf.keras.Model(inputs=[num_input, text_input], outputs=out)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # ---------- 10) 콜백 ----------
    log_dir = 'LogFile/Log_Model_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    tensorboard = TensorBoard(log_dir=log_dir)
    early_stop  = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)

    # ---------- 11) 학습 ----------
    train_inputs = {'num_input': X_train_num, 'text_input': X_train_text}
    val_inputs   = {'num_input': X_val_num,   'text_input': X_val_text}

    history = model.fit(
        train_inputs, y_train,
        validation_data=(val_inputs, y_val),
        batch_size=32, epochs=50,
        callbacks=[early_stop, tensorboard],
        verbose=1
    )

    # ---------- 12) 평가 ----------
    test_inputs = {'num_input': X_test_num, 'text_input': X_test_text}
    test_loss, test_acc = model.evaluate(test_inputs, y_test, verbose=0)
    print(f"[TEST] loss={test_loss:.4f}, acc={test_acc:.4f}, period> {val_end.date()}")

    # ---------- 13) 저장 ----------
    # 새 모델 이름으로 저장 권장 (기존 예측 코드와 구분)
    new_model_path = 'SaveModel/upstock_model_v2.keras'
    model.save(new_model_path)

    # 토크나이저 저장
    try:
        with open('SaveModel/upstock_tokenizer_v2.pickle', 'wb') as f:
            pickle.dump(tokenizer, f)
        print('tokenizer v2 saved')
    except Exception as e:
        print('tokenizer 저장 실패:', e)


# else:
#     print('Model file is not exists, Start DeepLearning')
    
#     # Part Preprocessing
#     news_data = news_data.dropna(subset=['date'])
#     news_data['title'] = news_data['title'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
#     # utc 시간 고려
#     news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce', utc=True)
#     news_data['date'] = news_data['date'].dt.tz_localize(None).dt.date

#     unique_text = news_data['title'].tolist()
#     # 문자열 합침
#     unique_text = ''.join(unique_text)
#     # set 중복 제거 후 리스트 변환
#     unique_text = list(set(unique_text))
#     # 유니코드 순으로 나열
#     unique_text.sort()

#     # char level true 글자 단위, OOV 관례 : 나중에 추가되는 정규식에 없는 글자 정의
#     tokenizer = Tokenizer(char_level=False, oov_token='<OOV>')

#     # price data의 date 가져와서 merge
#     price_data['date'] = pd.to_datetime(price_data['date']).dt.date
#     merged = pd.merge(news_data, price_data[['date', 'Close', 'High', 'Low', 'Open', 'Volume', 'label']], on='date', how='inner')

#     # merged.to_csv('DataSets/Preprocessing.csv', index=False)

#     # 글자 치환
#     # 1. 타이틀을 리스트로
#     titles = merged['title'].tolist()
#     # 2. 타이틀 fit on text
#     tokenizer.fit_on_texts(titles)
#     # TODO TOKENIZER 개수
#     print(len(tokenizer.index_word))
#     # 3. 타이틀 text to sequences
#     titles = tokenizer.texts_to_sequences(titles)
#     # 4. pad sequences
#     titles = pad_sequences(titles, maxlen=110)
#     chart = merged[['Low', 'High', 'Open', 'Close', 'Volume']]
#     labels = np.array(merged['label'])

#     # TODO 이거 필요없으면 지우던가 EXCEPT 처리
#     # print(merged)
#     # exit()

#     # text, chart, label 데이터 쪼개기 0.2
#     X_train_text, X_val_text, X_train_chart, X_val_chart, y_train, y_val = train_test_split(
#         titles, chart, labels, test_size=0.2, random_state=42
#     )

#     # nomalization, 전체 데이터에서 하나의 평균과 분산을 사용
#     low_preprocessing = tf.keras.layers.Normalization(axis=None)
#     low_preprocessing.adapt(np.array(merged['Low']))
#     high_preprocessing = tf.keras.layers.Normalization(axis=None)
#     high_preprocessing.adapt(np.array(merged['High']))
#     open_preprocessing = tf.keras.layers.Normalization(axis=None)
#     open_preprocessing.adapt(np.array(merged['Open']))
#     close_preprocessing = tf.keras.layers.Normalization(axis=None)
#     close_preprocessing.adapt(np.array(merged['Close']))
#     volume_preprocessing = tf.keras.layers.Normalization(axis=None)
#     volume_preprocessing.adapt(np.array(merged['Volume']))

#     # nomalization result print
#     # normalized_close = close_preprocessing(np.array(merged['Close']))
#     # print(normalized_close.numpy())

#     # create input
#     low_input = tf.keras.Input(shape=(1, ), name='Low')
#     high_input = tf.keras.Input(shape=(1, ), name='High')
#     open_input = tf.keras.Input(shape=(1, ), name='Open')
#     close_input = tf.keras.Input(shape=(1, ), name='Close')
#     volume_input = tf.keras.Input(shape=(1, ), name='Volume')

#     x_low = low_preprocessing(low_input)
#     x_high = high_preprocessing(high_input)
#     x_open = open_preprocessing(open_input)
#     x_close = close_preprocessing(close_input)
#     x_volume = volume_preprocessing(volume_input)

#     # using functional api
#     model_input = tf.keras.Input(shape=(110,), name='model_input')
#     embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(model_input)
#     bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(embedding)
#     maxpool1d = tf.keras.layers.GlobalMaxPool1D()(bidirectional)
#     concat_layer = tf.keras.layers.Concatenate()([x_low, x_high, x_open, x_close, x_volume, maxpool1d])

#     dense1 = tf.keras.layers.Dense(64, activation='relu')(concat_layer)
#     dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
#     dense2 = tf.keras.layers.Dense(64, activation='relu')(dropout1)
#     dropout2 = tf.keras.layers.Dropout(0.3)(dense2)
#     dense3 = tf.keras.layers.Dense(32, activation='relu')(dropout2)
#     model_output = tf.keras.layers.Dense(1, activation='sigmoid')(dense3)

#     model = tf.keras.Model(inputs=[model_input, low_input, high_input, open_input, close_input, volume_input], outputs=model_output)
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#     train_inputs = {
#         'model_input' : X_train_text,
#         'Low' : np.array(X_train_chart['Low']).reshape(-1, 1),
#         'High' : np.array(X_train_chart['High']).reshape(-1, 1),
#         'Open' : np.array(X_train_chart['Open']).reshape(-1, 1),
#         'Close' : np.array(X_train_chart['Close']).reshape(-1, 1),
#         'Volume' : np.array(X_train_chart['Volume']).reshape(-1, 1),
#     }

#     val_inputs = {
#         'model_input': X_val_text,
#         'Low': np.array(X_val_chart['Low']).reshape(-1, 1),
#         'High': np.array(X_val_chart['High']).reshape(-1, 1),
#         'Open': np.array(X_val_chart['Open']).reshape(-1, 1),
#         'Close': np.array(X_val_chart['Close']).reshape(-1, 1),
#         'Volume': np.array(X_val_chart['Volume']).reshape(-1, 1),
#     }

#     # Part callback | tensorboard --logdir=LogFile/
#     # time.time() 큰 숫자가 최신
#     #TODO 나중에 모델 딥러닝할 때 적용시킬 것 : datetime.datetime.now().strftime("%Y-%m-%d_%H-%M") 
#     # tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_Model_' + str(int(time.time()))) )
#     tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_Model_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) )
#     early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

#     # train
#     model.fit(train_inputs, y_train, validation_data=(val_inputs, y_val), batch_size=32, epochs=50, callbacks=[early_stop, tensorboard])
#     model.summary()
#     model.save(model_path)
#     # 비상용
#     model.save(model_path_h5)

#     try:
#         with open(tokenizer_path, 'wb') as f:
#             pickle.dump(tokenizer, f)
#     except Exception as e:
#         print('tokenizer 저장 실패')