import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, Dropout, LayerNormalization, \
    Add
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


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
        self.query_dense = tf.keras.layers.Dense(embed_dim)
        self.key_dense = tf.keras.layers.Dense(embed_dim)
        self.value_dense = tf.keras.layers.Dense(embed_dim)
        self.combine_heads = tf.keras.layers.Dense(embed_dim)

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
    """Pre-Layer Normalization을 사용하는 Transformer Encoder Block."""

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
    """ Transformer 모델을 생성, 컴파일, 학습합니다."""
    config, data = self.config, self.data
    n_timesteps, n_features = data.X_train.shape[1], data.X_train.shape[2]
    embed_dim = config['model_embed_dim']
    inputs = Input(shape=(n_timesteps, n_features))

    x = Dense(embed_dim)(inputs)
    x = PositionalEncoding(position=n_timesteps, d_model=embed_dim)(x)

    for _ in range(config['transformer_num_blocks']):
        x = TransformerEncoderBlock(
            embed_dim=embed_dim,
            num_heads=config['mha_num_heads'],
            ff_dim=config['transformer_ff_dim'],
            rate=config['dropout_rate']
        )(x)

    x = GlobalAveragePooling1D()(x)
    x = Dropout(config['dropout_rate'])(x)
    x = Dense(config['latent_dim'], activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)

    self.model = KerasModel(inputs=inputs, outputs=outputs)

    optimizer = AdamW(learning_rate=config['learning_rate'], weight_decay=config['weight_decay'])
    self.model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    self.model.summary()

    print("\n=== Transformer 모델 학습 시작 ===")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1, min_lr=1e-6)
    ]

    self.history = self.model.fit(
        data.X_train, data.y_train,
        epochs=config['epochs'], batch_size=config['batch_size'],
        validation_data=(data.X_test, data.y_test),
        callbacks=callbacks,
        class_weight=self.config.get('class_weight'),
        verbose=2
    )

    self.data.y_pred_proba = self.model.predict(data.X_test).flatten()
    print("=== Transformer 모델 학습 완료 ===")