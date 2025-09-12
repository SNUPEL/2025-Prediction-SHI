import numpy as np
import pandas as pd
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall
from tensorflow.keras.layers import (Input, ConvLSTM2D, BatchNormalization, Flatten, Dense, Activation,
                                     Dropout, TimeDistributed, Add, Conv1D, Conv3D, LSTM, Bidirectional)
from tensorflow.keras.callbacks import EarlyStopping
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
    X_valid_3d = np.array(list(self.data.df_x_valid_matrix_dict.values()))
    y_valid = np.array(list(self.data.df_y_valid_dict.values()))
    X_test_3d = np.array(list(self.data.df_x_test_matrix_dict.values()))

    X_train_5d = create_convlstm_5d_sequences_internal(X_train_3d, self.config['sub_window_size'])
    X_valid_5d = create_convlstm_5d_sequences_internal(X_valid_3d, self.config['sub_window_size'])
    X_test_5d = create_convlstm_5d_sequences_internal(X_test_3d, self.config['sub_window_size'])

    valid_set = (X_valid_5d, y_valid)

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

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        verbose=1,
        restore_best_weights=True
    )

    fit_params = self.config['fit_parameter'].copy()
    fit_params['callbacks'] = [early_stopping_callback]

    history = self.model.fit(x=X_train_5d, y=y_train, validation_data=valid_set, **fit_params)

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
