"""
Config.py

dataclasses를 이용함
env 파일을 로드하고 supabase 및 각 파일의 path를 담고있음
"""

import os
import logging
import sys

from dotenv import load_dotenv
from supabase import create_client, Client
from dataclasses import dataclass   # create data class

# env load
load_dotenv()

"""
log ex)
2025-09-18 00:09:10,835 [INFO] upstock.nodes.predict - [positive] Investors haven't been this bullish on stocks in months : 0.92
"""
logging.basicConfig(
    level=logging.INFO, # info 이상 log print
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    stream=sys.stdout   # import sys 터미널에 로그 출력
)

@dataclass(frozen=True) # 불변
class SupabaseConfig:
    url: str = os.getenv('SUPABASE_URL')
    key: str = os.getenv('SUPABASE_KEY')
    bucket_name: str = 'sentiment_file' # download model file
    
    @property
    def client(self) -> Client:
        return create_client(self.url, self.key)

@dataclass(frozen=True)
class PathConfig:
    sentiment_data: str = 'DataSets/upstock-sentiment-data.csv' # sentiment data
    tokenizer: str = 'SaveModel/upstock_sentiment_tokenizer.pickle'
    model: str = 'SaveModel/upstock_sentiment_model.keras'
    model_h5: str = 'SaveModel/upstock_sentiment_model.h5'
    history: str = 'SaveModel/upstock_sentiment_pkl.pkl' # import matplotlib.pyplot as plt

# export
supabase = SupabaseConfig().client
paths = PathConfig()