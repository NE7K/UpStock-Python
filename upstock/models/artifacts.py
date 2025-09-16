import pickle
import os
import pandas as pd

from tensorflow.keras.models import load_model

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

# load tokenizer => TextVectorization로 변경 가능성 유의
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
