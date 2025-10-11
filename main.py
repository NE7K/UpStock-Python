"""
Main entry point
"""
# compatibility issue import
# from keras.preprocessing.text import Tokenizer
# from keras.callbacks import EarlyStopping
# from keras.preprocessing.sequence import pad_sequences
# from keras.callbacks import TensorBoard

# pip github connect | pip freeze > piplist.txt
from upstock.storage.downloader import download_model_file
from upstock.builders.pipeline import run_pipeline

if __name__ == '__main__':
    """
    download model file : supabase model and tokenizer download
    run pipeline : connect predict, train node 
    """
    # download_model_file()
    run_pipeline()