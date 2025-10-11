"""
Training node for 3-class sentiment model (Softmax + StratifiedKFold + Class Weight)
"""

import os, time, pickle, logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from upstock.config import paths
from upstock.models.artifacts import load_csv

logger = logging.getLogger(__name__)

def run_train():
    # TEST sentiment data regex
    # sentiment_data['Text'] = sentiment_data['Text'].str.replace('[^a-zA-Z0-9$%+\- ]', '', regex=True)
    
    sentiment_data = load_csv(paths.sentiment_data, 'Sentiment dataset')
    if sentiment_data is None:
        logger.error('Sentiment dataset not available')
        return

    # Text / Label
    texts = sentiment_data['Text'].astype(str).tolist()
    labels = sentiment_data['Sentiment'].astype(int).values # int로 불러옴

    # Tokenize
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=300, padding='post', truncating='post') # 300 이후 뒷 글자 자름
    
    """
    ===== 문장 길이 통계 =====
    평균 길이: 112.49
    중간값: 97
    90% 백분위: 201
    95% 백분위: 240
    99% 백분위: 346
    최대 길이: 2107
    """

    # Class weight (불균형 보정)
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = dict(zip(classes, weights))
    logger.info(f"Class weights: {class_weights}")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    all_reports = []

    for train_idx, val_idx in skf.split(padded, labels):
        logger.info(f"Fold {fold} / 5")

        X_train, X_val = padded[train_idx], padded[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Build model
        model_input = tf.keras.Input(shape=(300,), name='model_input')
        embedding = tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=128)(model_input)
        bidirectional = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(embedding)
        maxpool1d = tf.keras.layers.GlobalMaxPooling1D()(bidirectional)
        dense1 = tf.keras.layers.Dense(64, activation='relu')(maxpool1d)
        dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
        output = tf.keras.layers.Dense(3, activation='softmax')(dropout1)
        model = tf.keras.Model(inputs=model_input, outputs=output)

        # Compile (Softmax + class weight)
        model.compile(
            loss='sparse_categorical_crossentropy',
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
            metrics=['accuracy']
        )
        
        # INFO callback | tensorboard --logdir=LogFile/
        # time.time() 큰 숫자가 최신
        # TODO 가독성 좋지 못하면 아래로 대체
        # tensorboard = TensorBoard(log_dir='LogFile/Log{}'.format('_SentimentModel_' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")))
        log_dir = f'LogFile/Log_Softmax_Fold{fold}_{int(time.time())}'
        callbacks = [
            TensorBoard(log_dir=log_dir),
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=2  # 1이 진행바
        )
        
        # model.summary()

        # 예측 및 혼동행렬
        y_pred = np.argmax(model.predict(X_val), axis=1)
        cm = confusion_matrix(y_val, y_pred)
        report = classification_report(y_val, y_pred, target_names=['negative', 'neutral', 'positive'], output_dict=True)
        all_reports.append(report)

        # Confusion Matrix 시각화 저장
        os.makedirs('LogFile/ConfMatrix', exist_ok=True)
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['neg','neu','pos'],
                    yticklabels=['neg','neu','pos'])
        plt.title(f'Confusion Matrix - Fold {fold}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'ConfMatrix/fold_{fold}.png', bbox_inches='tight')
        plt.close()

        fold += 1 # fold number ++

    # 평균 리포트
    avg_acc = np.mean([r['accuracy'] for r in all_reports])
    logger.info(f"Average Accuracy (5-Fold): {avg_acc:.4f}")

    # Save final model + tokenizer + history
    try:
        model.save(paths.model)
        model.save(paths.model_h5)
        with open(paths.tokenizer, 'wb') as f:
            pickle.dump(tokenizer, f)
        with open(paths.history, 'wb') as f:
            pickle.dump(history.history, f)
    except Exception as e:
        logger.error(f"Save failed : {e}")

    logger.info("Training completed successfully")
