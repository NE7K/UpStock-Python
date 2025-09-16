import os
from dotenv import load_dotenv
from supabase import create_client, Client

# env load
load_dotenv()

supabase_url = os.getenv('SupaBase_Url')
supabase_key = os.getenv('SupaBase_Key')

# connect sb sdk
supabase: Client = create_client(
    supabase_url,
    supabase_key,
)

# 경로
sentiment_path = 'DataSets/upstock-sentiment-data.csv' # sentiment data
tokenizer_path = 'SaveModel/upstock_sentiment_tokenizer.pickle'
model_path = 'SaveModel/upstock_sentiment_model.keras'
model_path_h5 = 'SaveModel/upstock_sentiment_model.h5' # compatibility issue .h5

# 논문 근거
model_pkl_path = 'SaveModel/upstock_sentiment_pkl.pkl' # import matplotlib.pyplot as plt
