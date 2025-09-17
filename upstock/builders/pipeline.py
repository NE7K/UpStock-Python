"""
training or predict
"""

import os
from upstock.config import paths
from upstock.nodes.predict import run_predict
from upstock.nodes.train import run_train

def run_pipeline():
    """
    save model exists => predict
    save model not exists => DeepLearning
    """
    if os.path.exists(paths.model) and os.path.exists(paths.tokenizer):
        run_predict()
    else:
        print('Sentiment Model and Tokenizer is not exists, Start DeepLearning')
        run_train()