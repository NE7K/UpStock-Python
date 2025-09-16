import os
import time
import pickle
import datetime
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from sklearn.model_selection import train_test_split

from upstock.config import sentiment_path, model_path, tokenizer_path, model_pkl_path
from upstock.models.artifacts import load_file

def run_trian():
    # TEST sentiment data regex
    # sentiment_data['Text'] = sentiment_data['Text'].str.replace('[^a-zA-Z0-9$%+\- ]', '', regex=True)

    sentiment_data = load_file(sentiment_path)

    # PART word level
    sentiment_text = sentiment_data['Text'].astype(str).tolist()
    label = sentiment_data['Sentiment'].values

    # Tokenizer
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(sentiment_text)
    sequences = tokenizer.texts_to_sequences(sentiment_text)
    sentiment_text = pad_sequences(sequences, maxlen=141)

    # text, sentiment data split 0.2
    X_train, X_val, y_train, y_val = train_test_split(sentiment_text, label, test_size=0.2, random_state=42)

    # using functional api
    model_input = tf.keras.Input(shape=(141,), name='model_input')
    embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(model_input)
    bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(embedding)
    maxpool1d = tf.keras.layers.GlobalMaxPool1D()(bidirectional)
    dense1 = tf.keras.layers.Dense(64, activation='relu')(maxpool1d)
    dropout1 = tf.keras.layers.Dropout(0.3)(dense1) # overfitting 
    dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout1) # 추가 데이터 확보시 Added dense layer add
    model_output = tf.keras.layers.Dense(1, activation='sigmoid')(dense2)
    
    model = tf.keras.Model(inputs=model_input, outputs=model_output)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # INFO callback | tensorboard --logdir=LogFile/
    # time.time() 큰 숫자가 최신
    # TODO 가독성 좋지 못하면 아래로 대체
    # tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_SentimentModel_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
    tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_SentimentModel_' + str(int(time.time()))) )
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1) # early stop alarm

    # history => pkl 
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=20,
        callbacks=[early_stop, tensorboard]
    )

    # model.summary()
    
    # save
    try:
        model.save(model_path)
        # model.save(model_path_h5) # h5 version
    except Exception as e:
        print(f'model save fail : {e}')

    try:
        with open(tokenizer_path, 'wb') as f:
            pickle.dump(tokenizer, f)
    except Exception as e:
            print(f'tokenizer save fail : {e}')

    try:
        with open(model_pkl_path, 'wb') as f:
            pickle.dump(history.history, f) # matplot
    except Exception as e:
            print(f'model pkl save fail : {e}')
            
            
