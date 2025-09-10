# -*- coding: utf-8 -*-
# test.py — 저장된 모델 입력 자동 매핑 + 추론단 뉴스 비중 가중 옵션 포함
# 기존 학습 분기는 거의 그대로 두고, 추론에서만 숫자 스케일을 조정해 뉴스 영향도를 올릴 수 있게 함.

import numpy as np
import pandas as pd
import pickle
import os
import re
import datetime

# validation x > split Test Data
from sklearn.model_selection import train_test_split

# recent keras / tf-keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

# =========================
# 경로/상수
# =========================
price_path = 'DataSets/stock_price_data.csv'
news_path = 'DataSets/analyst_ratings_processed.csv'
tokenizer_path = 'SaveModel/upstock_tokenizer.pickle'
model_path = 'SaveModel/upstock_model.keras'
model_path_h5 = 'SaveModel/upstock_model.h5'

MAXLEN = 110

# 추론 시 숫자 피처(OHLCV) 비중 (1.0이 원래값). 낮추면 뉴스 영향 ↑
NUM_WEIGHT = 0.6   # 필요시 0.5, 0.4로 낮춰가며 감도 확인. 원복은 1.0

# =========================
# Keras 커스텀 표준화 함수 등록
# (과거 TextVectorization(standardize=custom_standardize_fn) 저장본 호환)
# =========================
try:
    from keras.saving import register_keras_serializable
except Exception:
    from tensorflow.keras.utils import register_keras_serializable as register_keras_serializable

@register_keras_serializable(package="Custom")
def custom_standardize_fn(x):
    if not isinstance(x, tf.Tensor):
        x = tf.convert_to_tensor(x, dtype=tf.string)
    x = tf.strings.lower(x)
    # 영문/숫자/공백/주요 기호만 유지: $ % - . , & '
    x = tf.strings.regex_replace(x, r"[^a-z0-9\$\%\-\.\,&'\s]", " ")
    x = tf.strings.regex_replace(x, r"\s+", " ")
    return tf.strings.strip(x)

CUSTOM_OBJECTS = {"custom_standardize_fn": custom_standardize_fn}

# =========================
# 공통 로드 유틸
# =========================
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

def check_all_model(path, description):
    if os.path.exists(path):
        try:
            model = load_model(path, custom_objects=CUSTOM_OBJECTS)
            print(f'{description} load complete')
            return model
        except Exception as e:
            print(f'{description} load fail : {e}')
            return None
    else:
        print(f'{description} not exists')
        return None

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

# =========================
# (핵심) 모델 입력 자동 매핑 어댑터
#  - 저장된 모델의 input 이름/개수/dtype을 보고
#    우리가 가진 데이터(title + OHLCV)를 맞춰 dict로 구성
# =========================
def _keras_tensor_info(t):
    """KerasTensor에서 (name_without_port, dtype, shape) 추출"""
    name = t.name.split(':')[0] if hasattr(t, 'name') else 'input'
    dtype = tf.as_dtype(t.dtype) if hasattr(t, 'dtype') else None
    shape = tuple(t.shape.as_list()) if hasattr(t, 'shape') else None
    return name, dtype, shape

def build_predict_feed(model, title_text, ohlcv, tokenizer=None):
    """
    model: 로드된 keras 모델
    title_text: str (뉴스 제목)
    ohlcv: dict with keys Low, High, Open, Close, Volume (float)
    tokenizer: (선택) 정수 시퀀스용 토크나이저
    반환: 모델이 기대하는 입력 딕셔너리
    """
    # 기본 OHLCV 벡터 (순서: Low, High, Open, Close, Volume)
    num_vec = np.array([
        ohlcv.get('Low', 0.0),
        ohlcv.get('High', 0.0),
        ohlcv.get('Open', 0.0),
        ohlcv.get('Close', 0.0),
        ohlcv.get('Volume', 0.0),
    ], dtype=np.float32).reshape(1, -1)

    # >>> 추론단 가중: 숫자 비중 낮춰 뉴스 상대 가중 ↑
    if NUM_WEIGHT != 1.0:
        num_vec = num_vec * float(NUM_WEIGHT)

    inputs = model.inputs if hasattr(model, 'inputs') else []
    in_infos = [_keras_tensor_info(t) for t in inputs]
    in_names = [n for (n, _, _) in in_infos]
    in_map = {}

    # 1) 대표 케이스: ['text','num'] (문자열 + 수치벡터 한 덩어리)
    if set(in_names) == set(['text', 'num']):
        for name, dtype, shape in in_infos:
            if name == 'text':
                if dtype == tf.string:
                    in_map['text'] = np.array([title_text], dtype=object)
                else:
                    # 정수 시퀀스 기대
                    if tokenizer is not None:
                        seq = tokenizer.texts_to_sequences([title_text])
                        in_map['text'] = pad_sequences(seq, maxlen=MAXLEN)
                    else:
                        in_map['text'] = np.zeros((1, MAXLEN), dtype=np.int32)
            elif name == 'num':
                width = shape[-1] if shape is not None and len(shape) >= 2 else num_vec.shape[1]
                if width is None:
                    width = num_vec.shape[1]
                v = num_vec
                if v.shape[1] < width:
                    pad = np.zeros((1, width - v.shape[1]), dtype=np.float32)
                    v = np.concatenate([v, pad], axis=1)
                elif v.shape[1] > width:
                    v = v[:, :width]
                in_map['num'] = v
        return in_map

    # 2) 예전 구조: 'model_input' + 개별 Low/High/Open/Close/Volume
    if 'model_input' in in_names and all(k in in_names for k in ['Low','High','Open','Close','Volume']):
        # model_input이 정수 시퀀스일 가능성 높음
        # 혹시 문자열 기대면 밑에서 dtype 확인
        if tokenizer is not None:
            seq = tokenizer.texts_to_sequences([title_text])
            in_map['model_input'] = pad_sequences(seq, maxlen=MAXLEN)
        else:
            # dtype 체크
            for name, dtype, _ in in_infos:
                if name == 'model_input' and dtype == tf.string:
                    in_map['model_input'] = np.array([title_text], dtype=object)
                    break
            else:
                in_map['model_input'] = np.zeros((1, MAXLEN), dtype=np.int32)

        l, h, o, c, v = num_vec.squeeze().tolist()
        in_map['Low'] = np.array([[l]], dtype=np.float32)
        in_map['High'] = np.array([[h]], dtype=np.float32)
        in_map['Open'] = np.array([[o]], dtype=np.float32)
        in_map['Close'] = np.array([[c]], dtype=np.float32)
        in_map['Volume'] = np.array([[v]], dtype=np.float32)
        return in_map

    # 3) 문자열 단일 입력만 있는 경우 (내부 TextVectorization)
    if len(in_names) == 1:
        name, dtype, shape = in_infos[0]
        if dtype == tf.string:
            return {name: np.array([title_text], dtype=object)}
        else:
            # 정수 시퀀스 기대
            if tokenizer is not None:
                seq = tokenizer.texts_to_sequences([title_text])
                return {name: pad_sequences(seq, maxlen=MAXLEN)}
            else:
                return {name: np.zeros((1, MAXLEN), dtype=np.int32)}

    # 4) 숫자 단일 입력만 있는 경우 (예외적; 안전패스)
    if len(in_names) == 1 and in_names[0] not in ['text','model_input']:
        name, dtype, shape = in_infos[0]
        width = shape[-1] if shape is not None and len(shape) >= 2 else num_vec.shape[1]
        if width is None:
            width = num_vec.shape[1]
        v = num_vec
        if v.shape[1] < width:
            pad = np.zeros((1, width - v.shape[1]), dtype=np.float32)
            v = np.concatenate([v, pad], axis=1)
        elif v.shape[1] > width:
            v = v[:, :width]
        return {name: v}

    # 5) 마지막 안전장치: 입력 순서대로 매핑 시도
    feed = {}
    used_num = False
    for name, dtype, shape in in_infos:
        if dtype == tf.string and not name.lower().startswith(('low','high','open','close','volume')):
            feed[name] = np.array([title_text], dtype=object)
        else:
            if not used_num:
                width = shape[-1] if shape is not None and len(shape) >= 2 else num_vec.shape[1]
                if width is None:
                    width = num_vec.shape[1]
                v = num_vec
                if v.shape[1] < width:
                    pad = np.zeros((1, width - v.shape[1]), dtype=np.float32)
                    v = np.concatenate([v, pad], axis=1)
                elif v.shape[1] > width:
                    v = v[:, :width]
                feed[name] = v
                used_num = True
            else:
                feed[name] = np.zeros((1, shape[-1] if shape and len(shape)>=2 and shape[-1] else 1), dtype=np.float32)
    return feed

# =========================
# 디버그: 뉴스/차트 감도 비교
# =========================
def predict_debug(model, title, ohlcv, tokenizer=None):
    feed_base = build_predict_feed(model, title, ohlcv, tokenizer)
    p_base = float(model.predict(feed_base, verbose=0).squeeze())

    feed_text = build_predict_feed(model, title, {'Low':0,'High':0,'Open':0,'Close':0,'Volume':0}, tokenizer)
    p_textonly = float(model.predict(feed_text, verbose=0).squeeze())

    feed_num = build_predict_feed(model, "", ohlcv, tokenizer)
    p_numonly = float(model.predict(feed_num, verbose=0).squeeze())

    print(f"[DEBUG] base={p_base:.4f} | text_only={p_textonly:.4f} | num_only={p_numonly:.4f}")
    return p_base, p_textonly, p_numonly

# =========================
# 데이터셋 로드
# =========================
price_data = load_file(price_path, 'stock price csv file')
news_data  = load_file(news_path,  'stock news csv file')

# 모델 / 토크나이저
model     = check_all_model(model_path, 'model .keras')
model_h5  = check_all_model(model_path_h5, 'model .h5')  # TextVectorization 포함 모델은 h5 미지원 경고가 정상
tokenizer = load_pickle(tokenizer_path, 'tokenizer')

# =========================
# 예측 경로 (모델/토크나이저 존재)
# =========================
if os.path.exists(model_path) and os.path.exists(tokenizer_path):
    # 재확인 로드
    model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    # 예시 입력 (원하는 문장으로 바꿔도 됨)
    raw_title = 'S&P 500, Nasdaq recover from losses as markets bet on September rate cut'
    ohlcv = {
        'Low':    23475.330078,
        'High':   23860.25,
        'Open':   23841.980469,
        'Close':  23652.439453,
        'Volume': 8413730000,
    }

    feed = build_predict_feed(model, raw_title, ohlcv, tokenizer)
    prediction = model.predict(feed)
    y = float(np.array(prediction).squeeze())
    print(prediction, 'up' if y >= 0.5 else 'down')

    # 디버그(선택): 뉴스/차트 감도 확인
    predict_debug(model, raw_title, ohlcv, tokenizer)

    # 벤치(선택): 뉴스만 넣고 문구 반응 확인
    titles = [
        "Fed hints at aggressive rate cuts; stocks rally",
        "Surging unemployment rate sparks recession fears",
        "Apple announces record iPhone sales, shares jump",
        "Geopolitical tensions escalate; risk-off sentiment grows",
    ]
    for t in titles:
        feed_t = build_predict_feed(model, t, {'Low':0,'High':0,'Open':0,'Close':0,'Volume':0}, tokenizer)
        p = float(model.predict(feed_t, verbose=0).squeeze())
        print(f"{p:.4f}  |  {t}")

else:
    print('Model file is not exists, Start DeepLearning')

    # =========================
    # (아래는 기존 학습 분기: 원본 로직 유지)
    # =========================

    # Part Preprocessing
    news_data = news_data.dropna(subset=['date'])
    news_data['title'] = news_data['title'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    # utc 시간 고려
    news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce', utc=True)
    news_data['date'] = news_data['date'].dt.tz_localize(None).dt.date

    unique_text = news_data['title'].tolist()
    unique_text = ''.join(unique_text)
    unique_text = list(set(unique_text))
    unique_text.sort()

    # char level true 글자 단위, OOV 관례 : 나중에 추가되는 정규식에 없는 글자 정의
    tokenizer = Tokenizer(char_level=False, oov_token='<OOV>')

    # price data의 date 가져와서 merge
    price_data['date'] = pd.to_datetime(price_data['date']).dt.date
    merged = pd.merge(
        news_data,
        price_data[['date', 'Close', 'High', 'Low', 'Open', 'Volume', 'label']],
        on='date', how='inner'
    )

    # 글자 치환
    titles = merged['title'].tolist()
    tokenizer.fit_on_texts(titles)
    print(len(tokenizer.word_index))
    titles = tokenizer.texts_to_sequences(titles)
    titles = pad_sequences(titles, maxlen=MAXLEN)
    chart = merged[['Low', 'High', 'Open', 'Close', 'Volume']]
    labels = np.array(merged['label'])

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
    model_input = tf.keras.Input(shape=(MAXLEN,), name='model_input')
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

    model = tf.keras.Model(inputs=[model_input, low_input, high_input, open_input, close_input, volume_input],
                           outputs=model_output)
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
    os.makedirs('LogFile', exist_ok=True)
    tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_Model_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) )
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

    # train
    model.fit(train_inputs, y_train, validation_data=(val_inputs, y_val),
              batch_size=1024, epochs=5, callbacks=[early_stop, tensorboard])
    model.summary()
    os.makedirs('SaveModel', exist_ok=True)
    model.save(model_path)      # Keras v3 포맷
    model.save(model_path_h5)   # h5는 TextVectorization 포함 모델에서 경고 발생 가능(무시해도 됨)

    try:
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
    except Exception as e:
        print(f'tokenizer save fail : {e}')
