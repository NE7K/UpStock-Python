# -*- coding: utf-8 -*-
# test.py — 저장된 모델 입력 자동 매핑
# + 뉴스 비중 60%(text 60 / num 40)
# + 텍스트 온도보정(분산 확대)
# + 금융 감성(lexicon) 보정
# + ★배치 테스트★: 샘플별로 "뉴스 헤드라인 + 시나리오별 OHLCV(상승/하락)"를 묶어서 예측

import numpy as np
import pandas as pd
import pickle
import os
import re
import datetime

from sklearn.model_selection import train_test_split

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

# (옵션) 추론 시 숫자 피처(OHLCV) 비중 (1.0=원래). 낮출수록 뉴스 영향 상대적↑
NUM_WEIGHT = 0.6

# 추론 앙상블: base vs (text-only/num-only 혼합)
# 뉴스 비중 60%로 고정하려면 base 비중 0, text:0.6 / num:0.4
TEXT_BOOST = 1.0   # base 비중 0%
ALPHA_TEXT = 0.6   # text-only 60% / num-only 40%

# 텍스트 온도 보정(0.7~0.5 추천, 낮출수록 확률 분산 확대/차별화↑)
TEXT_TEMP = 0.7

# 금융 감성 보정 강도 (로짓 공간 가중). 1.0~3.0 권장
FIN_SENTI_WEIGHT = 2.0

# =========================
# 배치 테스트 시나리오 설정
# - 각 샘플은 'title'과 함께 'scenario'('up'|'down') 또는 직접 'ohlcv' dict 제공 가능
# - 'scenario'만 주면 아래 make_ohlcv()로 자동 생성
# =========================
ANCHOR_CLOSE = 23652.439453  # 기준가(임의). 필요시 바꾸세요.
BASE_VOLUME  = 8_413_730_000 # 기준 거래량(임의)

def make_ohlcv(anchor_close=ANCHOR_CLOSE, direction='up', pct=2.0, volume=BASE_VOLUME):
    """
    간단한 OHLCV 생성기:
      - direction='up'  : 대략 +pct% 상승 마감
      - direction='down': 대략 -pct% 하락 마감
    """
    ac = float(anchor_close)
    p  = float(pct) / 100.0
    if direction == 'up':
        open_  = ac * (1 - 0.002)          # 살짝 낮게 출발
        close_ = ac * (1 + p)               # +pct% 상승 마감
        high_  = max(open_, close_) * 1.003
        low_   = min(open_, close_) * 0.997
        vol_   = volume * 1.05              # 상승일 때 거래량 약간↑ (임의)
    else:
        open_  = ac * (1 + 0.002)           # 살짝 높게 출발
        close_ = ac * (1 - p)               # -pct% 하락 마감
        high_  = max(open_, close_) * 1.003
        low_   = min(open_, close_) * 0.997
        vol_   = volume * 1.08              # 하락일 때 거래량 약간↑ (임의)
    return {
        'Low':    float(low_),
        'High':   float(high_),
        'Open':   float(open_),
        'Close':  float(close_),
        'Volume': float(vol_),
    }

# 배치 테스트 샘플 (원하는 대로 수정/추가하세요)
BATCH_SAMPLES = [
    # 긍정 뉴스 + 상승 시나리오 OHLCV
    {"group": "positive", "title": "Fed hints at aggressive rate cuts; stocks rally", "scenario": "up"},
    {"group": "positive", "title": "Apple announces record iPhone sales, shares jump", "scenario": "up"},

    # 부정 뉴스 + 하락 시나리오 OHLCV
    {"group": "negative", "title": "Surging unemployment rate sparks recession fears", "scenario": "down"},
    {"group": "negative", "title": "Geopolitical tensions escalate; risk-off sentiment grows", "scenario": "down"},

    # 필요시 직접 OHLCV를 박아넣을 수 있음 (scenario 무시)
    # {"group": "custom", "title": "Custom headline example", "ohlcv": {"Low":..., "High":..., "Open":..., "Close":..., "Volume":...}}
]

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

    # 추론단 숫자 가중 조정 (뉴스 상대 비중↑)
    if NUM_WEIGHT != 1.0:
        num_vec = num_vec * float(NUM_WEIGHT)

    inputs = getattr(model, 'inputs', [])
    in_infos = [_keras_tensor_info(t) for t in inputs]
    in_names = [n for (n, _, _) in in_infos]
    in_map = {}

    # 1) ['text','num'] (문자열 + 수치벡터 한 덩어리)
    if set(in_names) == set(['text', 'num']):
        for name, dtype, shape in in_infos:
            if name == 'text':
                if dtype == tf.string:
                    in_map['text'] = np.array([title_text], dtype=object)
                else:
                    if tokenizer is not None:
                        seq = tokenizer.texts_to_sequences([title_text])
                        in_map['text'] = pad_sequences(seq, maxlen=MAXLEN)
                    else:
                        in_map['text'] = np.zeros((1, MAXLEN), dtype=np.int32)
            elif name == 'num':
                width = shape[-1] if shape is not None and len(shape) >= 2 else num_vec.shape[1]
                if width is None: width = num_vec.shape[1]
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
        if tokenizer is not None:
            seq = tokenizer.texts_to_sequences([title_text])
            in_map['model_input'] = pad_sequences(seq, maxlen=MAXLEN)
        else:
            for name, dtype, _ in in_infos:
                if name == 'model_input' and dtype == tf.string:
                    in_map['model_input'] = np.array([title_text], dtype=object)
                    break
            else:
                in_map['model_input'] = np.zeros((1, MAXLEN), dtype=np.int32)

        l, h, o, c, v = num_vec.squeeze().tolist()
        in_map['Low']    = np.array([[l]], dtype=np.float32)
        in_map['High']   = np.array([[h]], dtype=np.float32)
        in_map['Open']   = np.array([[o]], dtype=np.float32)
        in_map['Close']  = np.array([[c]], dtype=np.float32)
        in_map['Volume'] = np.array([[v]], dtype=np.float32)
        return in_map

    # 3) 문자열 단일 입력
    if len(in_names) == 1:
        name, dtype, shape = in_infos[0]
        if dtype == tf.string:
            return {name: np.array([title_text], dtype=object)}
        else:
            if tokenizer is not None:
                seq = tokenizer.texts_to_sequences([title_text])
                return {name: pad_sequences(seq, maxlen=MAXLEN)}
            else:
                return {name: np.zeros((1, MAXLEN), dtype=np.int32)}

    # 4) 숫자 단일 입력 (예외)
    if len(in_names) == 1 and in_names[0] not in ['text','model_input']:
        name, dtype, shape = in_infos[0]
        width = shape[-1] if shape is not None and len(shape) >= 2 else num_vec.shape[1]
        if width is None: width = num_vec.shape[1]
        v = num_vec
        if v.shape[1] < width:
            pad = np.zeros((1, width - v.shape[1]), dtype=np.float32)
            v = np.concatenate([v, pad], axis=1)
        elif v.shape[1] > width:
            v = v[:, :width]
        return {name: v}

    # 5) 마지막 안전장치
    feed = {}
    used_num = False
    for name, dtype, shape in in_infos:
        if dtype == tf.string and not name.lower().startswith(('low','high','open','close','volume')):
            feed[name] = np.array([title_text], dtype=object)
        else:
            if not used_num:
                width = shape[-1] if shape is not None and len(shape) >= 2 else num_vec.shape[1]
                if width is None: width = num_vec.shape[1]
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
# 금융 감성(lexicon) 보정 유틸
# =========================
FIN_POS_WORDS = {
    "rally","rallies","surge","surges","soar","soars","jump","jumps",
    "beat","beats","record","strong","gain","gains","rebound","rises","rise",
    "upgrade","upgrades","outperform","optimism","bullish","buyback","stimulus",
}
FIN_NEG_WORDS = {
    "fear","fears","recession","unemployment","miss","misses","tension","tensions",
    "geopolitical","conflict","riskoff","selloff","sell-off","sell","plunge","plunges",
    "fall","falls","drop","drops","downgrade","downgrades","concern","concerns",
    "weak","slump","strike","bankruptcy","lawsuit","default","war","escalate","escalates",
    "inflation","headwind","headwinds"
}
FIN_POS_PHRASES = [
    r"rate cut(s)?",
    r"record (revenue|sales|earnings)",
    r"better than expected",
    r"beats (estimates|expectations)",
]
FIN_NEG_PHRASES = [
    r"rate hike(s)?",
    r"surging unemployment",
    r"risk[-\s]?off",
    r"worse than expected",
    r"miss(es)? (estimates|expectations)",
    r"geopolitical tensions?",
]
NEGATIONS = {"not","no","without","less","fewer","decline","declines","declined","declining"}

_word_re = re.compile(r"[a-z]+")
def _tokenize_lower(text: str):
    text = text.lower()
    text = text.replace("risk-off", "riskoff")
    return _word_re.findall(text)

def fin_sentiment_score(text: str, window: int = 2) -> float:
    if not text:
        return 0.0
    t = text.lower()
    toks = _tokenize_lower(t)
    pos = neg = 0
    for i, w in enumerate(toks):
        score = 0
        if w in FIN_POS_WORDS:
            score = 1
        elif w in FIN_NEG_WORDS:
            score = -1
        if score != 0:
            start = max(0, i - window)
            if any(n in toks[start:i] for n in NEGATIONS):
                score *= -1
            if score > 0: pos += 1
            elif score < 0: neg += 1
    for pat in FIN_POS_PHRASES:
        if re.search(pat, t):
            pos += 1
    for pat in FIN_NEG_PHRASES:
        if re.search(pat, t):
            neg += 1
    if pos == 0 and neg == 0:
        return 0.0
    s = (pos - neg) / float(pos + neg)
    return max(-1.0, min(1.0, s))

# =========================
# TVEC 점검(옵션)
# =========================
def get_textvec(model):
    for lyr in model.layers:
        if lyr.__class__.__name__ == 'TextVectorization':
            return lyr
    return None

def debug_textvec(model):
    vec = get_textvec(model)
    if vec is None:
        print("[TVEC] TextVectorization layer not found.")
        return
    vocab = vec.get_vocabulary()
    print(f"[TVEC] vocab_size={len(vocab)} | head={vocab[:20]}")

# =========================
# 디버그/앙상블
# =========================
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p/(1-p))

def _sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def predict_debug(model, title, ohlcv, tokenizer=None):
    feed_base = build_predict_feed(model, title, ohlcv, tokenizer)
    p_base = float(model.predict(feed_base, verbose=0).squeeze())
    feed_text = build_predict_feed(model, title, {'Low':0,'High':0,'Open':0,'Close':0,'Volume':0}, tokenizer)
    p_textonly = float(model.predict(feed_text, verbose=0).squeeze())
    feed_num = build_predict_feed(model, "", ohlcv, tokenizer)
    p_numonly = float(model.predict(feed_num, verbose=0).squeeze())
    print(f"[DEBUG] base={p_base:.4f} | text_only={p_textonly:.4f} | num_only={p_numonly:.4f}")
    return p_base, p_textonly, p_numonly

def predict_with_text_boost(model, title, ohlcv, tokenizer=None,
                            text_boost: float = TEXT_BOOST, alpha_text: float = ALPHA_TEXT):
    """
    p_final = (1-text_boost)*p_base + text_boost*(alpha_text*p_text_polar + (1-alpha_text)*p_num)
    - text_boost=1.0, alpha_text=0.6 → 뉴스 60%, 차트 40% (base 0%)
    - p_text_polar: 텍스트 확률에 온도 보정 + 금융 감성 보정 적용
    """
    # base
    feed_base = build_predict_feed(model, title, ohlcv, tokenizer)
    p_base = float(model.predict(feed_base, verbose=0).squeeze())
    # text-only
    feed_text = build_predict_feed(model, title, {'Low':0,'High':0,'Open':0,'Close':0,'Volume':0}, tokenizer)
    p_text = float(model.predict(feed_text, verbose=0).squeeze())
    # num-only
    feed_num = build_predict_feed(model, "", ohlcv, tokenizer)
    p_num = float(model.predict(feed_num, verbose=0).squeeze())
    # 텍스트 온도 보정
    p_text_adj = _sigmoid(_logit(p_text) / float(TEXT_TEMP))
    # 금융 감성(−1~+1) 로짓 가감
    s_text = fin_sentiment_score(title)
    p_text_polar = _sigmoid(_logit(p_text_adj) + float(FIN_SENTI_WEIGHT) * s_text)

    p_mix = alpha_text * p_text_polar + (1.0 - alpha_text) * p_num
    p_final = (1.0 - text_boost) * p_base + text_boost * p_mix

    return {
        'p_base': p_base,
        'p_text': p_text,
        'p_text_adj': p_text_adj,
        's_text': s_text,
        'p_text_polar': p_text_polar,
        'p_num': p_num,
        'p_mix': p_mix,
        'p_final': p_final
    }

# =========================
# 배치 실행 유틸
# =========================
def ensure_ohlcv(sample):
    """샘플에 ohlcv가 없으면 scenario로 자동 생성"""
    if 'ohlcv' in sample and isinstance(sample['ohlcv'], dict):
        return sample['ohlcv']
    direction = sample.get('scenario', 'up')
    return make_ohlcv(ANCHOR_CLOSE, direction=direction, pct=2.0, volume=BASE_VOLUME)

def run_batch(model, tokenizer, samples):
    """샘플 리스트를 순회하며 결과를 보기 좋게 출력"""
    # 그룹별 묶음 출력
    by_group = {}
    for s in samples:
        g = s.get('group', 'default')
        by_group.setdefault(g, []).append(s)

    for g, items in by_group.items():
        print(f"\n=== GROUP: {g.upper()} ({len(items)} samples) ===")
        for idx, s in enumerate(items, 1):
            title = s['title']
            ohlcv = ensure_ohlcv(s)
            res = predict_with_text_boost(model, title, ohlcv, tokenizer)
            # 핵심 수치 요약
            verdict = "up" if res['p_final'] >= 0.5 else "down"
            print(f"{idx:02d}. {verdict.upper()}  "
                  f"final={res['p_final']:.4f} | mix={res['p_mix']:.4f} | "
                  f"text={res['p_text']:.4f}→adj={res['p_text_adj']:.4f}→polar={res['p_text_polar']:.4f} "
                  f"(s={res['s_text']:+.3f}) | num={res['p_num']:.4f} | "
                  f"OHLCV[O={ohlcv['Open']:.1f} H={ohlcv['High']:.1f} L={ohlcv['Low']:.1f} C={ohlcv['Close']:.1f} V={ohlcv['Volume']:.0f}] "
                  f"| {title}")

# =========================
# 데이터셋 로드
# =========================
price_data = load_file(price_path, 'stock price csv file')
news_data  = load_file(news_path,  'stock news csv file')

# 모델 / 토크나이저
model     = check_all_model(model_path, 'model .keras')
model_h5  = check_all_model(model_path_h5, 'model .h5')  # 일부 모델은 h5 저장 미지원일 수 있음(경고 무시 가능)
tokenizer = load_pickle(tokenizer_path, 'tokenizer')

# =========================
# 예측 경로 (모델/토크나이저 존재)
# =========================
if os.path.exists(model_path) and os.path.exists(tokenizer_path):
    model = load_model(model_path, custom_objects=CUSTOM_OBJECTS)

    # (옵션) TVEC 점검
    debug_textvec(model)

    # 단일 예시(원래 하던 방식)
    demo_title = 'S&P 500, Nasdaq recover from losses as markets bet on September rate cut'
    demo_ohlcv = make_ohlcv(ANCHOR_CLOSE, direction='up', pct=2.0, volume=BASE_VOLUME)

    feed = build_predict_feed(model, demo_title, demo_ohlcv, tokenizer)
    prediction = model.predict(feed)
    y = float(np.array(prediction).squeeze())
    print(prediction, 'up' if y >= 0.5 else 'down')
    predict_debug(model, demo_title, demo_ohlcv, tokenizer)

    boosted = predict_with_text_boost(model, demo_title, demo_ohlcv, tokenizer)
    print(f"[NEWS60] base={boosted['p_base']:.4f} | text={boosted['p_text']:.4f} "
          f"| text_adj={boosted['p_text_adj']:.4f} | s_text={boosted['s_text']:+.3f} "
          f"| text_polar={boosted['p_text_polar']:.4f} | num={boosted['p_num']:.4f} "
          f"| mix={boosted['p_mix']:.4f} | final={boosted['p_final']:.4f} -> "
          f"{'up' if boosted['p_final']>=0.5 else 'down'}")

    # ★배치 테스트: 긍정 뉴스+상승 OHLCV / 부정 뉴스+하락 OHLCV 묶어서 검증
    run_batch(model, tokenizer, BATCH_SAMPLES)

else:
    print('Model file is not exists, Start DeepLearning')

    # =========================
    # (아래는 기존 학습 분기: 원본 로직 유지)
    # =========================

    # Part Preprocessing
    news_data = news_data.dropna(subset=['date'])
    news_data['title'] = news_data['title'].str.replace('[^a-zA-Z0-9 ]', '', regex=True)
    news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce', utc=True)
    news_data['date'] = news_data['date'].dt.tz_localize(None).dt.date

    unique_text = news_data['title'].tolist()
    unique_text = ''.join(unique_text)
    unique_text = list(set(unique_text))
    unique_text.sort()

    tokenizer = Tokenizer(char_level=False, oov_token='<OOV>')

    price_data['date'] = pd.to_datetime(price_data['date']).dt.date
    merged = pd.merge(
        news_data,
        price_data[['date', 'Close', 'High', 'Low', 'Open', 'Volume', 'label']],
        on='date', how='inner'
    )

    titles = merged['title'].tolist()
    tokenizer.fit_on_texts(titles)
    print(len(tokenizer.word_index))
    titles = tokenizer.texts_to_sequences(titles)
    titles = pad_sequences(titles, maxlen=MAXLEN)
    chart = merged[['Low', 'High', 'Open', 'Close', 'Volume']]
    labels = np.array(merged['label'])

    X_train_text, X_val_text, X_train_chart, X_val_chart, y_train, y_val = train_test_split(
        titles, chart, labels, test_size=0.2, random_state=42
    )

    low_preprocessing = tf.keras.layers.Normalization(axis=None); low_preprocessing.adapt(np.array(merged['Low']))
    high_preprocessing = tf.keras.layers.Normalization(axis=None); high_preprocessing.adapt(np.array(merged['High']))
    open_preprocessing = tf.keras.layers.Normalization(axis=None); open_preprocessing.adapt(np.array(merged['Open']))
    close_preprocessing = tf.keras.layers.Normalization(axis=None); close_preprocessing.adapt(np.array(merged['Close']))
    volume_preprocessing = tf.keras.layers.Normalization(axis=None); volume_preprocessing.adapt(np.array(merged['Volume']))

    low_input = tf.keras.Input(shape=(1, ), name='Low')
    high_input = tf.keras.Input(shape=(1, ), name='High')
    open_input = tf.keras.Input(shape=(1, ), name='Open')
    close_input = tf.keras.Input(shape=(1, ), name='Close')
    volume_input = tf.keras.Input(shape=(1, ), name='Volume')

    x_low   = low_preprocessing(low_input)
    x_high  = high_preprocessing(high_input)
    x_open  = open_preprocessing(open_input)
    x_close = close_preprocessing(close_input)
    x_volume= volume_preprocessing(volume_input)

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

    os.makedirs('LogFile', exist_ok=True)
    tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_Model_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")) )
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

    model.fit(train_inputs, y_train, validation_data=(val_inputs, y_val), batch_size=32, epochs=50, callbacks=[early_stop, tensorboard])
    model.summary()
    os.makedirs('SaveModel', exist_ok=True)
    model.save(model_path)      # Keras v3 포맷
    model.save(model_path_h5)   # 일부 구성에서 경고 가능(무시)
    try:
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
    except Exception as e:
        print(f'tokenizer save fail : {e}')
