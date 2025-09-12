import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, GlobalAveragePooling1D, Dropout, LayerNormalization, Add)
from tensorflow.keras.optimizers import AdamW
# from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.metrics import Recall
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight


class CustomMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"임베딩 차원({embed_dim})은 헤드 수({num_heads})로 나누어 떨어져야 합니다."
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=None):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position, self.d_model = position, d_model
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2], angle_rads[:, 1::2] = np.sin(angle_rads[:, 0::2]), np.cos(angle_rads[:, 1::2])
        self.pos_encoding = tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def get_angles(self, position, i, d_model):
        return position * (1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model)))

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class TransformerEncoderBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.att = CustomMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="elu"), Dense(embed_dim)])
        self.layernorm1, self.layernorm2 = LayerNormalization(epsilon=1e-6), LayerNormalization(epsilon=1e-6)
        self.dropout1, self.dropout2 = Dropout(rate), Dropout(rate)

    def call(self, inputs, training=False):
        ln_output1 = self.layernorm1(inputs)
        attn_output = self.att(ln_output1)
        out1 = Add()([inputs, self.dropout1(attn_output, training=training)])
        ln_output2 = self.layernorm2(out1)
        ffn_output = self.ffn(ln_output2)
        return Add()([out1, self.dropout2(ffn_output, training=training)])


def get_Transformer(self):
    """ Transformer 모델을 생성, 컴파일, 학습하고 예측합니다. (EarlyStopping 적용) """
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

    # (batch, features, time) -> (batch, time, features)
    X_train = X_train.transpose(0, 2, 1).astype('float32')
    y_train = y_train.astype('float32')
    X_valid = X_valid.transpose(0, 2, 1).astype('float32')
    y_valid = y_valid.astype('float32')
    X_test = X_test.transpose(0, 2, 1).astype('float32')

    valid_set = (X_valid, y_valid)

    input_shape = X_train.shape[1:]
    model_params = self.config['model_parameter']

    # FFN 차원(ff_dim)을 embed_dim의 4배로 자동 설정
    model_params['ff_dim'] = model_params['embed_dim'] * 4

    input_layer = Input(shape=input_shape)
    x = Dense(model_params['embed_dim'])(input_layer)
    x = PositionalEncoding(position=input_shape[0], d_model=model_params['embed_dim'])(x)
    for _ in range(model_params['num_blocks']):
        x = TransformerEncoderBlock(embed_dim=model_params['embed_dim'], num_heads=model_params['num_heads'],
                                    ff_dim=model_params['ff_dim'], rate=model_params['dropout_rate'])(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(model_params['dropout_rate'])(x)
    output_layer = Dense(**model_params['output_layer'])(x)
    self.model = Model(inputs=input_layer, outputs=output_layer)

    compile_params = self.config['compile_parameter']
    optimizer = AdamW(learning_rate=compile_params['learning_rate'],
                      weight_decay=compile_params.get('weight_decay', 0.01))
    self.model.compile(optimizer=optimizer, loss=compile_params['loss'], metrics=compile_params['metrics'])
    self.model.summary()

    # EarlyStopping 사용 여부에 따라 콜백 리스트를 설정
    callbacks = []
    if self.config.get('use_early_stopping', False):
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ))

    fit_params = self.config['fit_parameter'].copy()
    fit_params['callbacks'] = callbacks

    # class weight 자동으로 계산하는 코드 추가
    if isinstance(fit_params.get('class_weight'), str) and fit_params['class_weight'].lower() == 'auto':
        y_labels = self.data.df_y_train['label'].values  # 훈련 데이터의 실제 레이블을 가져옵니다.
        classes = np.unique(y_labels)
        weights = compute_class_weight('balanced', classes=classes, y=y_labels)
        class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
        fit_params['class_weight'] = class_weight_dict
        print(f"자동 계산된 class_weight: {class_weight_dict}")

    # 콜백이 포함된 'fit_params'를 사용하여 모델 학습
    history = self.model.fit(x=X_train, y=y_train, validation_data=valid_set, **fit_params)

    self.history = {'train_loss': history.history['loss'], 'val_loss': history.history['val_loss']}

    y_pred_proba = self.model.predict(X_test)
    if y_pred_proba.shape[-1] == 1:
        y_pred_proba = y_pred_proba.flatten()
    y_pred_class = (y_pred_proba > self.config['threshold']).astype(int)
    self.data.df_y_pred = pd.DataFrame(y_pred_class, index=self.data.name_test, columns=['label'])