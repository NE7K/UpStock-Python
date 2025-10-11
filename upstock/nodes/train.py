"""
Training node for 3-class sentiment model (Softmax + StratifiedKFold + Class Weight + Best-Fold Save)
"""

import os, time, pickle, logging, shutil
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
    labels = sentiment_data['Sentiment'].astype(int).values  # int로 불러옴

    # 전역 Tokenizer 및 pad_sequences 생성 제거
    # tokenizer = Tokenizer(oov_token="<OOV>")
    # tokenizer.fit_on_texts(texts)
    # seqs = tokenizer.texts_to_sequences(texts)
    # padded = pad_sequences(seqs, maxlen=300, padding='post', truncating='post')

    """
    ===== 문장 길이 통계 =====
    평균 길이: 112.49
    중간값: 97
    90% 백분위: 201
    95% 백분위: 240
    99% 백분위: 346
    최대 길이: 2107
    """

    # Class weight
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    class_weights = dict(zip(classes, weights))
    logger.info(f"Class weights : {class_weights}")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    all_reports, all_histories = [], {}

    # Best Fold 저장용 변수
    best_fold, best_val_acc, best_model, best_tokenizer = None, 0.0, None, None

    # padded 대신 texts 직접 split
    for train_idx, val_idx in skf.split(texts, labels):
        logger.info(f"Fold {fold} / 5")

        # Fold별 텍스트 분리
        X_train_texts = [texts[i] for i in train_idx]
        X_val_texts   = [texts[i] for i in val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        # Fold별 Tokenizer 새로 학습
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train_texts)

        X_train_seq = tokenizer.texts_to_sequences(X_train_texts)
        X_val_seq   = tokenizer.texts_to_sequences(X_val_texts)

        X_train = pad_sequences(X_train_seq, maxlen=300, padding='post', truncating='post')
        X_val   = pad_sequences(X_val_seq, maxlen=300, padding='post', truncating='post')

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
        log_dir = f'LogFile/Log_Softmax_Fold{fold}'
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)

        callbacks = [
            TensorBoard(log_dir=log_dir),
            EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        ]

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=32,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1  # 1이 진행바
        )
        
        # model.summary()

        # 예측 및 혼동행렬
        y_pred = np.argmax(model.predict(X_val), axis=1)
        cm = confusion_matrix(y_val, y_pred)
        report = classification_report(
            y_val, y_pred,
            target_names=['negative', 'neutral', 'positive'],
            output_dict=True
        )
        all_reports.append(report)
        
        os.makedirs("SaveModel", exist_ok=True)
        with open(f'SaveModel/history_fold{fold}.pkl', 'wb') as f:
            pickle.dump(history.history, f)
            
        all_histories[f'fold_{fold}'] = history.history

        # [ADDED] 현재 fold의 최고 검증 정확도 추적
        val_acc = max(history.history['val_accuracy'])
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_fold = fold
            best_model = model
            best_tokenizer = tokenizer

        # Confusion Matrix 시각화 저장
        os.makedirs('Image/ConfMatrix', exist_ok=True)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['neg', 'neu', 'pos'],
                    yticklabels=['neg', 'neu', 'pos'])
        plt.title(f'Confusion Matrix - Fold {fold}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'Image/ConfMatrix/fold_{fold}.png', bbox_inches='tight')
        plt.close()

        fold += 1  # fold number ++

    # 평균 리포트
    avg_acc = np.mean([r['accuracy'] for r in all_reports])
    avg_loss = np.mean([np.mean(h['loss']) for h in all_histories.values()])
    logger.info(f"Average Accuracy (5-Fold) : {avg_acc:.4f}")
    logger.info(f"Average Training Loss : {avg_loss:.4f}")

    # Best fold 모델과 tokenizer 저장
    try:
        os.makedirs("SaveModel", exist_ok=True)
        best_model.save(os.path.join("SaveModel", f"best_model_fold{best_fold}.keras"))
        with open(os.path.join("SaveModel", f"best_tokenizer_fold{best_fold}.pkl"), 'wb') as f:
            pickle.dump(best_tokenizer, f)
        logger.info(f"Best fold : {best_fold}, val_acc : {best_val_acc:.4f}")
    except Exception as e:
        logger.error(f"Best model save failed : {e}")

    # 마지막 fold 모델 저장 제거 (best만 저장하도록)
    # try:
    #     model.save(paths.model)
    #     model.save(paths.model_h5)
    #     with open(paths.tokenizer, 'wb') as f:
    #         pickle.dump(tokenizer, f)
    #     with open(paths.history, 'wb') as f:
    #         pickle.dump(all_histories, f)
    # except Exception as e:
    #     logger.error(f"Save failed : {e}")
