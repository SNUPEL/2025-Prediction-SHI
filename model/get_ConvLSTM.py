import numpy as np
import pandas as pd
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall
from tensorflow.keras.layers import (Input, ConvLSTM2D, BatchNormalization, Flatten, Dense, Activation,
                                     Dropout, TimeDistributed, Add, Conv1D, Conv3D, LSTM, Bidirectional)
from tensorflow.keras.regularizers import l2
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False
})


def create_convlstm_5d_sequences_internal(data_array, sub_window_size):
    if data_array.ndim != 3:
        raise ValueError(f"입력 배열은 3차원이어야 함. 현재 형태: {data_array.shape}")

    # num_samples, num_features, timesteps 순서
    num_samples, num_features, timesteps_original = data_array.shape

    if sub_window_size > timesteps_original:
        raise ValueError(f"sub_window_size({sub_window_size})가 원본 타임스텝({timesteps_original})보다 큼")

    # 시간 축을 따라 슬라이딩 윈도우 적용
    new_timesteps = timesteps_original - sub_window_size + 1
    # 결과 형태: (샘플, 새 타임스텝, 특성, 윈도우 크기)
    sub_sequences_4d = np.zeros((num_samples, new_timesteps, num_features, sub_window_size))
    for i in range(num_samples):
        for j in range(new_timesteps):
            # 모든 특성에 대해 시간(세 번째 축)을 슬라이싱
            sub_sequences_4d[i, j, :, :] = data_array[i, :, j:j + sub_window_size]

    # 채널 차원 추가 -> (샘플, 새 타임스텝, 특성, 윈도우 크기, 1)
    final_5d_array = np.expand_dims(sub_sequences_4d, axis=-1)
    return final_5d_array


def get_ConvLSTM(self):
    tf.random.set_seed(self.config['random_state'])
    if self.config['undersampling'] or self.config['oversampling']:
        X_train_3d = np.array(list(self.data.df_x_train_matrix_dict_after_sampling.values()))
        y_train = np.array(list(self.data.df_y_train_dict_after_sampling.values()))
    else:
        X_train_3d = np.array(list(self.data.df_x_train_matrix_dict.values()))
        y_train = np.array(list(self.data.df_y_train_dict.values()))
    X_test_3d = np.array(list(self.data.df_x_test_matrix_dict.values()))

    X_train_5d = create_convlstm_5d_sequences_internal(X_train_3d, self.config['sub_window_size'])
    X_test_5d = create_convlstm_5d_sequences_internal(X_test_3d, self.config['sub_window_size'])

    input_shape = X_train_5d.shape[1:]
    input_layer = Input(shape=input_shape)
    x = input_layer

    model_params = self.config['model_parameter']

    for i, params in enumerate(model_params['convlstm_layers']):
        shortcut = x
        layer_parmas = params.copy()
        kernel_width = layer_parmas.pop('kernel_width')
        activation_func = layer_parmas.pop('activation', 'relu')

        x = ConvLSTM2D(kernel_size=(input_shape[1], kernel_width), **layer_parmas)(x)

        if shortcut.shape[-1] != x.shape[-1]:
            shortcut = Conv3D(filters=layer_parmas['filters'], kernel_size=1, padding='same')(shortcut)

        x = Add()([x, shortcut])
        x = Activation(activation_func)(x)
        x = Dropout(model_params['dropout_rate'])(x)

    x = Flatten()(x)
    x = Dropout(model_params['dropout_rate'])(x)  # 최종 분류기 전 Dropout

    ###################################
    # for i, params in enumerate(model_params['convlstm_layers']):
    #     # 현재 블록의 입력(shortcut으로 사용)
    #     shortcut = x
    #     layer_parmas = params.copy()
    #     # ConvLSTM 레이어 (기존과 동일)
    #     # kernel_width와 activation을 params에서 분리
    #     kernel_width = layer_parmas.pop('kernel_width')
    #     activation_func = layer_parmas.pop('activation', 'relu')  # 설정에 activation이 없으면 relu 기본 사용
    #
    #     x = ConvLSTM2D(kernel_size=(input_shape[1], kernel_width), **layer_parmas)(x)
    #
    #     # 프로젝션: 입력(shortcut)과 출력(x)의 채널(필터 수)이 다르면 차원을 맞춰줌
    #     if shortcut.shape[-1] != x.shape[-1]:
    #         shortcut = Conv3D(filters=layer_parmas['filters'], kernel_size=1, padding='same')(shortcut)
    #
    #     # 잔차 연결 (입력과 출력을 더함)
    #     x = Add()([x, shortcut])
    #     x = Activation(activation_func)(x)  # 활성화 함수는 Add 이후에 적용
    #     x = Dropout(model_params['dropout_rate'])(x)
    #
    #     # --- 2. 차원 축소 (기존과 동일) ---
    # x = TimeDistributed(Flatten())(x)
    #
    # # --- 3. Bidirectional LSTM 레이어 블록 with 잔차 연결 ---
    # for i, params in enumerate(model_params['bilstm_layers']):
    #     # 마지막 BiLSTM 레이어는 return_sequences=False 이므로 잔차 연결을 적용하지 않음
    #     layer_parmas = params.copy()
    #     if not layer_parmas.get('return_sequences', False) and i == len(model_params['bilstm_layers']) - 1:
    #         regularizer = l2(layer_parmas.pop('l2_reg'))
    #         x = Bidirectional(LSTM(kernel_regularizer=regularizer, **layer_parmas))(x)
    #         x = Dropout(model_params['dropout_rate'])(x)
    #         break
    #
    #     # 현재 블록의 입력(shortcut으로 사용)
    #     shortcut = x
    #
    #     # BiLSTM 레이어 (기존과 동일)
    #     regularizer = l2(layer_parmas.pop('l2_reg'))
    #     activation_func = layer_parmas.pop('activation', 'relu')
    #
    #     x = Bidirectional(LSTM(kernel_regularizer=regularizer, **layer_parmas))(x)
    #
    #     # 프로젝션: 입력(shortcut)과 출력(x)의 유닛 수가 다르면 차원을 맞춰줌
    #     if shortcut.shape[-1] != x.shape[-1]:
    #         # BiLSTM 출력 차원은 units * 2 이므로, 필터 수를 맞춰줌
    #         shortcut = Conv1D(filters=layer_parmas['units'] * 2, kernel_size=1, padding='same')(shortcut)
    #
    #     # 잔차 연결
    #     x = Add()([x, shortcut])
    #     x = Activation(activation_func)(x)
    #     x = Dropout(model_params['dropout_rate'])(x)
    ###################################


    #######################################
    # # ConvLSTM 레이어
    # for params in model_params['convlstm_layers']:
    #     kernel_width = params.pop('kernel_width')
    #     x = ConvLSTM2D(kernel_size=(input_shape[1], kernel_width), **params)(x)
    #     x = Dropout(model_params['dropout_rate'])(x)
    #
    # x = TimeDistributed(Flatten())(x)
    #
    # # Bidirectional LSTM 레이어
    # for params in model_params['bilstm_layers']:
    #     regularizer = l2(params.pop('l2_reg'))
    #     x = Bidirectional(LSTM(kernel_regularizer=regularizer, **params))(x)
    #     x = Dropout(model_params['dropout_rate'])(x)
    ################################

    output_layer = Dense(**model_params['output_layer'])(x)
    self.model = Model(inputs=input_layer, outputs=output_layer)

    compile_params = self.config['compile_parameter']
    optimizer = Adam(learning_rate=compile_params['learning_rate'], clipnorm=compile_params['clipnorm'])
    self.model.compile(optimizer=optimizer, loss=compile_params['loss'], metrics=compile_params['metrics'])
    self.model.summary()

    history = self.model.fit(x=X_train_5d, y=y_train, **self.config['fit_parameter'])
    self.history = {
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
    }

    y_pred_proba = self.model.predict(X_test_5d)
    y_pred_class = (y_pred_proba > self.config['threshold']).astype(int)

    self.data.df_y_pred_proba = pd.DataFrame(
        y_pred_proba,
        index=self.data.name_test,
        columns=['label']
    )
    self.data.df_y_pred = pd.DataFrame(
        y_pred_class,
        index=self.data.name_test,
        columns=['label']
    )
