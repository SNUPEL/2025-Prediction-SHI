import numpy as np
import pandas as pd
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
from keras.models import Model
from keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

tf.config.optimizer.set_experimental_options({
    'layout_optimizer': False
})


def get_AutoEncoder(self):
    tf.random.set_seed(self.config['random_state'])
    X_train_normal = self.data.df_x_train_flatten[self.data.df_y_train['label'] == 0]
    num_features = self.data.df_x_train_flatten.shape[1]

    # --- Encoder ---
    input_layer = Input(shape=(num_features,))
    encoder = input_layer
    for layer_params in self.config['model_parameter']['encoder_layers']:
        encoder = Dense(**layer_params)(encoder)
    # --- Latent Space ---
    latent_space = Dense(**self.config['model_parameter']['latent_space'])(encoder)
    # --- Decoder ---
    decoder = latent_space
    for layer_params in self.config['model_parameter']['decoder_layers']:
        decoder = Dense(**layer_params)(decoder)

    output_layer = Dense(num_features,
                         activation=self.config['model_parameter']['output_activation'])(decoder)

    self.model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=self.config['compile_parameter']['learning_rate'])
    self.model.compile(optimizer=optimizer, loss=self.config['compile_parameter']['loss'])

    self.model.fit(
        X_train_normal, X_train_normal,
        **self.config['fit_parameter']
    )

    reconstructed_data = self.model.predict(self.data.df_x_test_flatten)
    reconstruction_error = np.mean(np.square(self.data.df_x_test_flatten - reconstructed_data), axis=1)

    train_normal_recon = self.model.predict(X_train_normal)
    train_normal_error = np.mean(np.square(X_train_normal - train_normal_recon), axis=1)

    threshold = np.percentile(train_normal_error, self.config['threshold_percentile'])
    y_pred = (reconstruction_error > threshold).astype(int)

    self.data.df_y_pred = pd.DataFrame(
        y_pred,
        index=self.data.name_test,
        columns=['label']
    )
