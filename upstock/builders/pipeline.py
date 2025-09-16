import os
from upstock.config import model_path, tokenizer_path
from upstock.nodes.predict_node import run_predict
from upstock.nodes.train_node import run_trian

# save model exists => predict
# save model not exists => DeepLearning
def run_pipeline():
    if os.path.exists(model_path) and os.path.exists(tokenizer_path):
        run_predict()
    else:
        print('Sentiment Model and Tokenizer is not exists, Start DeepLearning')
        run_trian()