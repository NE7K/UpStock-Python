import tensorflow as tf
import pickle
from keras.models import load_model

from keras.preprocessing.text import Tokenizer


model_path = 'SaveModel/upstock_model.keras'
tokenizer_path = 'SaveModel/upstock_tokenizer.pickle'

model = load_model(model_path)

with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)
    
