import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import (Input, Conv2D, Dropout, Flatten, Dense,
                                     Add, Activation, BatchNormalization, GlobalAveragePooling2D)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Recall


def get_ResNet(self):
    tf.random.set_seed(self.config['random_state'])
    if self.config['undersampling'] or self.config['oversampling']:
        X_train = np.array(list(self.data.df_x_train_matrix_dict_after_sampling.values()))
        y_train = np.array(list(self.data.df_y_train_dict_after_sampling.values()))
    else:
        X_train = np.array(list(self.data.df_x_train_matrix_dict.values()))
        y_train = np.array(list(self.data.df_y_train_dict.values()))
    X_test = np.array(list(self.data.df_x_test_matrix_dict.values()))


    input_shape = X_train.shape[1:]
    input_layer = Input(shape=input_shape)
    x = input_layer

    model_params = self.config['model_parameter']

    for layer_params in model_params['cnn_layers']:
        shortcut = x
        x = Conv2D(filters=layer_params['filters'],
                   kernel_size=layer_params['kernel_size'],
                   padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(filters=layer_params['filters'],
                   kernel_size=layer_params['kernel_size'],
                   padding='same')(x)
        x = BatchNormalization()(x)

        if shortcut.shape[-1] != layer_params['filters']:
            shortcut = Conv2D(filters=layer_params['filters'],
                              kernel_size=1,
                              padding='same')(shortcut)
            shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    x = Dropout(model_params['dropout_rate'])(x)
    output_layer = Dense(**model_params['output_layer'])(x)

    self.model = Model(inputs=input_layer, outputs=output_layer)

    compile_params = self.config['compile_parameter']
    optimizer = Adam(learning_rate=compile_params['learning_rate'], clipnorm=compile_params['clipnorm'])
    self.model.compile(optimizer=optimizer, loss=compile_params['loss'], metrics=compile_params['metrics'])
    self.model.summary()

    history = self.model.fit(x=X_train, y=y_train, **self.config['fit_parameter'])
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