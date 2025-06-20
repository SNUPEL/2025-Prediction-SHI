import numpy as np
import pandas as pd
import os
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow import keras
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall
from tensorflow.keras.layers import (Input, ConvLSTM2D, BatchNormalization, Flatten, Dense,
                                     Dropout, TimeDistributed, LSTM, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False
})


def create_convlstm_5d_sequences_internal(data_array, sub_window_size):
    if data_array.ndim != 3:
        raise ValueError(f"Train data 입력 배열 3차원이어야 함. 현재 형태: {data_array.shape}")

    num_samples, timesteps_original, num_features_inner = data_array.shape

    if sub_window_size > timesteps_original:
        raise ValueError(
            f"ConvLSTM 5D 시퀀스 생성 실패: sub_window_size({sub_window_size})가 원본 타임스텝({timesteps_original})보다 큼")

    # Sliding window 적용: 3D -> 4D
    # 결과: (num_samples, new_timesteps, window_size, num_features_inner)
    new_timesteps_for_convlstm = timesteps_original - sub_window_size + 1
    sub_sequences_4d = np.zeros((num_samples, new_timesteps_for_convlstm, sub_window_size, num_features_inner))
    for i in range(num_samples):
        for j in range(new_timesteps_for_convlstm):
            sub_sequences_4d[i, j, :, :] = data_array[i, j:j + sub_window_size, :]

    # 채널 차원 추가: 4D -> 5D
    # 결과: (num_samples, new_timesteps, window_size, num_features_inner, 1)
    final_5d_array = np.expand_dims(sub_sequences_4d, axis=-1)  # 마지막에 채널 1 추가

    return final_5d_array


def get_ConvLSTM(self):
    # 실제 데이터 (3D)를 5D ConvLSTM 입력 형태로 변환
    X_train_5d = create_convlstm_5d_sequences_internal(self.data.X_train, self.config['sub_window_size'])
    X_test_5d = create_convlstm_5d_sequences_internal(self.data.X_test, self.config['sub_window_size'])

    print(f"원본 훈련 데이터 형태 (3D): {self.data.X_train.shape}")
    print(f"변환된 훈련 데이터 형태 (5D): {X_train_5d.shape}")
    print(f"변환된 테스트 데이터 형태 (5D): {X_test_5d.shape}")

    # 데이터 셔플링
    X_shuffled, y_shuffled = shuffle(
        X_train_5d,  # 5D 변환된 데이터 사용
        self.data.y_train,
        random_state=self.config['random_state']
    )
    print("데이터 셔플 완료")

    # 훈련 데이터와 검증 데이터를 8:2 비율로 분리
    val_split_index = int(len(X_shuffled) * 0.8)
    X_train_part = X_shuffled[:val_split_index]
    y_train_part = y_shuffled[:val_split_index]
    X_val_part = X_shuffled[val_split_index:]
    y_val_part = y_shuffled[val_split_index:]
    print(f"훈련 데이터: {len(X_train_part)}개, 검증 데이터: {len(X_val_part)}개로 분리 완료.")

    # 클래스 가중치 계산
    class_weights = compute_class_weight(
        class_weight=self.config['class_weight'],
        classes=np.unique(y_train_part),
        y=y_train_part
    )
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    print(f"클래스 가중치 적용: {class_weight_dict}")

    # tf.data 파이프라인 생성
    batch_size = self.config.get('batch_size', 16)
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train_part, y_train_part))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val_part, y_val_part))
    val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

    # --- 모델 구성 및 컴파일 ---
    input_shape = X_train_5d.shape[1:]
    input_layer = Input(shape=input_shape, name='main_input')

    x = ConvLSTM2D(filters=32, kernel_size=(2, input_shape[2]), padding="same", return_sequences=True, activation='tanh')(
        input_layer)
    x = Dropout(0.3)(x)
    x = ConvLSTM2D(filters=16, kernel_size=(2, input_shape[2]), padding="same", return_sequences=True, activation='tanh')(x)
    x = Dropout(0.3)(x)
    x = TimeDistributed(Flatten())(x)
    x = Bidirectional(LSTM(32, return_sequences=True, activation='tanh', kernel_regularizer=l2(0.001)))(
        x)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(16, return_sequences=False, activation='tanh', kernel_regularizer=l2(0.001)))(
        x)
    x = Dropout(0.3)(x)
    x = Dense(16, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = Dropout(0.3)(x)
    output_layer = Dense(1, activation='sigmoid', name='output')(x)

    self.model = KerasModel(inputs=input_layer, outputs=output_layer)

    optimizer = Adam(learning_rate=0.001, clipnorm=1.0)
    self.model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy', Recall()])
    self.model.summary()

    # --- 모델 학습 ---
    print("\n=== 모델 학습 시작 ===")

    # early_stopping = EarlyStopping(monitor='val_loss', patience=self.config.get('patience', 10), verbose=1)
    early_stopping = EarlyStopping(monitor='val_recall', patience=self.config.get('patience', 10), mode='max',
                                   verbose=1)
    terminate_on_nan = TerminateOnNaN()
    callbacks_list = [early_stopping, terminate_on_nan]

    checkpoint_path = ""
    if self.config.get('save_model', False):
        checkpoint_path = os.path.join(self.config['result_folder_path'], 'best_convlstm_model.keras')
        print(f"모델 저장 활성화")
        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)
        callbacks_list.append(model_checkpoint)
    else:
        print("모델 저장 비활성화")

    history = self.model.fit(
        train_dataset,
        epochs=self.config.get('epochs', 50),
        validation_data=val_dataset,
        callbacks=callbacks_list,
        class_weight=class_weight_dict,
        verbose=1
    )
    print("=== 모델 학습 완료 ===")

    if self.config.get('save_model', False) and checkpoint_path and os.path.exists(checkpoint_path):
        print(f"가장 성능이 좋았던 모델 호출.")
        self.model = keras.models.load_model(checkpoint_path, compile=True)

        # --- 4. 모델 예측 ---
    print("\n=== 테스트 데이터 예측 시작 ===")

    test_dataset = tf.data.Dataset.from_tensor_slices(X_test_5d).batch(
        batch_size
    )

    y_pred_proba = self.model.predict(test_dataset)
    y_pred_class = (y_pred_proba > 0.5).astype(int)

    self.data.y_pred = y_pred_class.flatten()
    self.data.y_pred_proba = y_pred_proba.flatten()

    print("=== 예측 완료 ===")
