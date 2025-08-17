import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization,
                                     Add, Permute, GlobalAveragePooling1D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def get_MLP_Mixer(self):
    tf.random.set_seed(self.config['random_state'])

    if self.config.get('undersampling') or self.config.get('oversampling'):
        X_train = np.array(list(self.data.df_x_train_matrix_dict_after_sampling.values()))
        y_train = np.array(list(self.data.df_y_train_dict_after_sampling.values()))
    else:
        X_train = np.array(list(self.data.df_x_train_matrix_dict.values()))
        y_train = np.array(list(self.data.df_y_train_dict.values()))
    X_valid = np.array(list(self.data.df_x_valid_matrix_dict.values()))
    y_valid = np.array(list(self.data.df_y_valid_dict.values()))
    X_test = np.array(list(self.data.df_x_test_matrix_dict.values()))

    valid_set = (X_valid, y_valid)

    input_shape = X_train.shape[1:]
    n_features, n_timesteps = input_shape

    input_layer = Input(shape=input_shape)
    x = input_layer

    model_params = self.config['model_parameter']
    hidden_dim = model_params['hidden_dim']

    # 입력 특성 차원을 모델의 hidden_dim으로 세팅
    # (B, F, T) -> (B, T, F)
    x = Permute((2, 1))(x)
    x = Dense(hidden_dim)(x)
    # (B, T, D) -> (B, D, T)
    x = Permute((2, 1))(x)

    for _ in range(model_params['n_mixer_layers']):
        # 시간축 믹싱 (Token-Mixing)
        shortcut_token = x
        x_norm = LayerNormalization()(x)
        x_mixed = Dense(model_params['token_mlp_dim'], activation='gelu')(x_norm)
        x_mixed = Dense(n_timesteps)(x_mixed)
        x = Add()([shortcut_token, x_mixed])

        # 특성축 믹싱 (Channel-Mixing)
        shortcut_channel = x
        x_transposed = Permute((2, 1))(x)
        x_norm = LayerNormalization()(x_transposed)
        x_mixed = Dense(model_params['channel_mlp_dim'], activation='gelu')(x_norm)
        x_mixed = Dense(hidden_dim)(x_mixed)
        x_restored = Permute((2, 1))(x_mixed)
        x = Add()([shortcut_channel, x_restored])

    # 최종 분류
    x = LayerNormalization()(x)
    x = GlobalAveragePooling1D(data_format='channels_first')(x)
    x = Dropout(model_params['dropout_rate'])(x)
    output_layer = Dense(**model_params['output_layer'])(x)

    self.model = Model(inputs=input_layer, outputs=output_layer)

    compile_params = self.config['compile_parameter']
    optimizer = Adam(learning_rate=compile_params['learning_rate'], clipnorm=compile_params.get('clipnorm', 1.0))
    self.model.compile(optimizer=optimizer, loss=compile_params['loss'],
                       metrics=compile_params.get('metrics', ['accuracy']))
    self.model.summary()

    history = self.model.fit(x=X_train, y=y_train, validation_data=valid_set, **self.config['fit_parameter'])
    self.history = {
        'train_loss': history.history['loss'],
        'val_loss': history.history['val_loss'],
    }

    y_pred_proba = self.model.predict(X_test)
    y_pred_class = (y_pred_proba > self.config['threshold']).astype(int)

    self.data.df_y_pred = pd.DataFrame(
        y_pred_class,
        index=self.data.name_test,
        columns=['label']
    )