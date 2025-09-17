"""
loading models, tokenizer, dataset

*using logging
"""

import pickle
import os
import pandas as pd
import logging

from tensorflow.keras.models import load_model

logger = logging.getLogger(__name__)

def load_csv(path: str, description: str = 'CSV'):
    if os.path.exists(path):
        try:
            logger.info(f'{description} loaded complete')
            return pd.read_csv(path)
        except Exception as e:
            logger.error(f'{description} exists but failed to load {e}')
            return None
    else:
        logger.warning(f'{description} not exists {path}')
        return None

# load save model : https://www.tensorflow.org/guide/keras/save_and_serialize?hl=ko#savedmodel_%ED%98%95%EC%8B%9D
# model load 지침
def load_model_safe(path: str, description: str = 'Model'):
    if os.path.exists(path):
        try:
            model = load_model(path)
            logger.info(f'{description} loaded complete')
            return model
        except Exception as e:
            logger.error(f'{description} exists but failed to load {e}')
            return None
    else:
        logger.warning(f'{description} not exists {path}')
        return None

# load tokenizer => TextVectorization로 변경 가능성 유의
def load_pickle(path: str, description: str = 'Pickle'):
    if os.path.exists(path):
        try:
            with open(path, 'rb') as f:
                tokenizer = pickle.load(f)
                logger.info(f'{description} loaded complete')
                return tokenizer
        except Exception as e:
            logger.error(f'{description} exsists but failed to load {e}')
            return None
    else:
        logger.warning(f'{description} not exists {path}')
        return None

