import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, GlobalAveragePooling1D, Dropout, LayerNormalization, Add, Layer)
from tensorflow.keras.optimizers import AdamW  # AdamW 옵티마이저 사용
from tensorflow.keras.metrics import Recall  # 평가 지표로 Recall 사용
from tensorflow.keras.callbacks import EarlyStopping  # 조기 종료 콜백
from sklearn.utils.class_weight import compute_class_weight  # 불균형 데이터 처리를 위한 클래스 가중치 계산


class CustomMultiHeadAttention(tf.keras.layers.Layer):
    """
    커스텀 Multi-Head Attention 레이어.
    입력 시퀀스 내의 여러 위치에 동시에 주의(attention)를 기울여,
    각 위치의 표현을 풍부하게 만드는 메커니즘.
    """

    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim  # 임베딩 차원 (벡터 크기)
        self.num_heads = num_heads  # 어텐션 헤드의 수
        if embed_dim % num_heads != 0:
            raise ValueError(f"임베딩 차원({embed_dim})은 헤드 수({num_heads})로 나누어 떨어져야 합니다.")

        self.projection_dim = embed_dim // num_heads  # 각 헤드별 차원
        # Query, Key, Value, 최종 출력을 위한 Dense 레이어
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        """ Scaled Dot-Product Attention 계산 """
        # 1. Query와 Key를 행렬 곱하여 score 계산
        score = tf.matmul(query, key, transpose_b=True)
        # 2. Key의 차원(dim_key)으로 나눠 스케일링 (sharpness 조절)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        # 3. Softmax를 적용하여 어텐션 가중치(weights) 계산
        weights = tf.nn.softmax(scaled_score, axis=-1)
        # 4. 가중치와 Value를 곱하여 최종 어텐션 출력(output) 계산
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        """
        입력 텐서를 여러 헤드로 분리하는 함수.
        (batch, seq_len, embed_dim) -> (batch, num_heads, seq_len, projection_dim)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs, training=None):
        """ 레이어의 정방향 연산(forward pass) """
        batch_size = tf.shape(inputs)[0]
        # 1. 입력으로부터 Q, K, V 생성
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        # 2. Q, K, V를 여러 헤드로 분리
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        # 3. 각 헤드별로 어텐션 계산
        attention, weights = self.attention(query, key, value)
        # 4. 어텐션 결과를 다시 합치기
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        # 5. 최종 Dense 레이어를 통과시켜 출력
        output = self.combine_heads(concat_attention)
        return output


class PositionalEncoding(tf.keras.layers.Layer):
    """
    위치 인코딩 레이어.
    RNN과 달리 순서 정보가 없는 트랜스포머에게 토큰의 상대적/절대적 위치 정보를 제공.
    Sin, Cos 함수를 사용하여 위치별로 고유한 값을 더해줌.
    """

    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        # 미리 위치 인코딩 행렬을 계산해둠
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # 짝수 인덱스는 sin
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # 홀수 인덱스는 cos
        self.pos_encoding = tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def get_angles(self, position, i, d_model):
        return position * (1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model)))

    def call(self, inputs):
        """ 입력 텐서에 위치 인코딩 값을 더해줌 """
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """
    트랜스포머의 인코더 블록.
    Multi-Head Attention과 Feed-Forward Network 두 개의 서브 레이어로 구성.
    각 서브 레이어 후에는 Residual Connection(Add)과 Layer Normalization이 적용됨.
    """

    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerEncoderBlock, self).__init__(**kwargs)
        self.att = CustomMultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)  # Multi-Head Attention
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="elu"), Dense(embed_dim)])  # Feed-Forward Network
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=False):
        # --- 첫 번째 서브 레이어: Multi-Head Attention ---
        ln_output1 = self.layernorm1(inputs)  # (Pre-LN)
        attn_output = self.att(ln_output1)
        # Residual Connection
        out1 = Add()([inputs, self.dropout1(attn_output, training=training)])

        # --- 두 번째 서브 레이어: Feed-Forward Network ---
        ln_output2 = self.layernorm2(out1)  # (Pre-LN)
        ffn_output = self.ffn(ln_output2)
        # Residual Connection
        return Add()([out1, self.dropout2(ffn_output, training=training)])


class AttentionPooling(Layer):
    """
    학습 가능한 가중치를 사용한 어텐션 풀링 레이어.
    GlobalAveragePooling이 모든 타임스텝을 동일한 가중치로 평균내는 것과 달리,
    중요한 타임스텝에 더 높은 가중치를 부여하여 정보를 압축
    """

    def __init__(self, **kwargs):
        super(AttentionPooling, self).__init__(**kwargs)
        # 각 타임스텝의 중요도(score)를 계산하기 위한 Dense 레이어
        self.score_dense = Dense(1, name='attention_score_dense')

    def call(self, inputs):
        # 1. 각 타임스텝별 중요도(score) 계산 (batch, time, features) -> (batch, time, 1)
        scores = self.score_dense(inputs)
        # 2. Softmax를 통과시켜 가중치(weight) 계산. 모든 타임스텝 가중치의 합은 1이 됨.
        weights = tf.nn.softmax(scores, axis=1)
        # 3. 입력과 가중치를 곱하여 가중 평균(Weighted Average) 계산
        weighted_inputs = inputs * weights
        # 4. 시간 축(axis=1)에 대해 모두 합산하여 최종 컨텍스트 벡터(context_vector) 생성
        context_vector = tf.reduce_sum(weighted_inputs, axis=1)
        return context_vector


def get_Transformer(self):
    """ Transformer 모델을 생성, 컴파일, 학습하고 예측하는 전체 과정을 담은 함수. """
    tf.random.set_seed(self.config['random_state'])  # 재현성을 위한 랜덤 시드 고정

    # --- 1. 데이터 준비 ---
    # 샘플링 옵션에 따라 사용할 학습 데이터 선택
    if self.config.get('undersampling') or self.config.get('oversampling'):
        X_train = np.array(list(self.data.df_x_train_matrix_dict_after_sampling.values()))
        y_train = np.array(list(self.data.df_y_train_dict_after_sampling.values()))
    else:
        X_train = np.array(list(self.data.df_x_train_matrix_dict.values()))
        y_train = np.array(list(self.data.df_y_train_dict.values()))

    X_valid = np.array(list(self.data.df_x_valid_matrix_dict.values()))
    y_valid = np.array(list(self.data.df_y_valid_dict.values()))
    X_test = np.array(list(self.data.df_x_test_matrix_dict.values()))

    # 데이터의 축을 (batch, features, time) -> (batch, time, features) 형태로 변경
    # TensorFlow의 RNN/Transformer 레이어들은 시간 축이 두 번째에 오는 것을 표준으로 함
    X_train = X_train.transpose(0, 2, 1).astype('float32')
    y_train = y_train.astype('float32')
    X_valid = X_valid.transpose(0, 2, 1).astype('float32')
    y_valid = y_valid.astype('float32')
    X_test = X_test.transpose(0, 2, 1).astype('float32')

    valid_set = (X_valid, y_valid)  # 검증 데이터셋
    input_shape = X_train.shape[1:]  # 모델의 입력 형태 (time, features)
    model_params = self.config['model_parameter']  # config 파일에서 모델 하이퍼파라미터 로드

    # --- 2. 모델 구축 (Functional API 사용) ---
    input_layer = Input(shape=input_shape)

    # 임베딩 레이어: 입력 피처 차원을 모델의 임베딩 차원(embed_dim)으로 변환
    x = Dense(model_params['embed_dim'])(input_layer)

    # 위치 인코딩 추가
    x = PositionalEncoding(position=input_shape[0], d_model=model_params['embed_dim'])(x)

    # 설정된 블록 수(num_blocks)만큼 Transformer 인코더 블록을 쌓음
    for _ in range(model_params['num_blocks']):
        x = TransformerEncoderBlock(embed_dim=model_params['embed_dim'], num_heads=model_params['num_heads'],
                                    ff_dim=model_params['ff_dim'], rate=model_params['dropout_rate'])(x)

    # 풀링(Pooling) 레이어: 시계열 정보를 하나의 벡터로 압축
    # x = GlobalAveragePooling1D()(x)  # <-- 기존의 단순 평균 방식
    x = AttentionPooling()(x)  # <-- 중요한 타임스텝에 가중치를 주는 방식으로 변경

    x = Dropout(model_params['dropout_rate'])(x)

    # 출력 레이어: 최종적으로 경영악화 확률을 예측 (이진 분류)
    output_layer = Dense(**model_params['output_layer'])(x)

    # 최종 모델 정의
    self.model = Model(inputs=input_layer, outputs=output_layer)

    # --- 3. 모델 컴파일 ---
    compile_params = self.config['compile_parameter']
    optimizer = AdamW(learning_rate=compile_params['learning_rate'],
                      weight_decay=compile_params.get('weight_decay', 0.01))
    self.model.compile(optimizer=optimizer, loss=compile_params['loss'], metrics=compile_params['metrics'])
    self.model.summary()  # 모델 구조 출력

    # --- 4. 모델 학습 ---
    # EarlyStopping 콜백 설정: 검증 손실(val_loss)이 개선되지 않으면 학습 조기 종료
    callbacks = []
    if self.config.get('use_early_stopping', False):
        callbacks.append(EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True))

    fit_params = self.config['fit_parameter'].copy()
    fit_params['callbacks'] = callbacks

    # 클래스 가중치 자동 계산: 레이블 불균형 문제를 완화하기 위해 소수 클래스에 더 높은 가중치를 부여
    if isinstance(fit_params.get('class_weight'), str) and fit_params['class_weight'].lower() == 'auto':
        y_labels = self.data.df_y_train['label'].values
        classes = np.unique(y_labels)
        weights = compute_class_weight('balanced', classes=classes, y=y_labels)
        class_weight_dict = {cls: weight for cls, weight in zip(classes, weights)}
        fit_params['class_weight'] = class_weight_dict
        print(f"자동 계산된 class_weight: {class_weight_dict}")

    # 모델 학습 실행
    history = self.model.fit(x=X_train, y=y_train, validation_data=valid_set, **fit_params)

    # 학습 과정(loss) 저장
    self.history = {'train_loss': history.history['loss'], 'val_loss': history.history['val_loss']}

    # --- 5. 모델 예측 ---
    # 테스트 데이터에 대한 예측 확률 계산
    y_pred_proba = self.model.predict(X_test)
    if y_pred_proba.shape[-1] == 1:
        y_pred_proba = y_pred_proba.flatten()

    # 설정된 임계값(threshold)을 기준으로 0 또는 1로 클래스 예측
    y_pred_class = (y_pred_proba > self.config['threshold']).astype(int)

    # 최종 예측 결과를 DataFrame으로 저장
    self.data.df_y_pred = pd.DataFrame(y_pred_class, index=self.data.name_test, columns=['label'])