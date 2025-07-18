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
    # 1. 모델 학습을 위한 데이터 준비: 정상(0) 데이터만 추출
    X_train_normal = self.data.X_train[self.data.y_train == 0]

    # 2. 모델 구조 정의 (Dense Layer 기반)
    num_features = self.data.X_train.shape[1]

    input_layer = Input(shape=(num_features,))

    # 인코더: 데이터를 압축
    encoder = Dense(1024, activation='relu')(input_layer)
    encoder = Dense(256, activation='relu')(encoder)
    # encoder = Dense(128, activation='relu')(encoder)
    encoder = Dense(64, activation='relu')(encoder)
    latent_space = Dense(16, activation='relu')(encoder)

    # 디코더: 데이터를 복원
    decoder = Dense(64, activation='relu')(latent_space)
    # decoder = Dense(128, activation='relu')(decoder)
    decoder = Dense(256, activation='relu')(decoder)
    decoder = Dense(1024, activation='relu')(decoder)
    output_layer = Dense(num_features, activation='linear')(decoder)

    # 오토인코더 모델 컴파일
    self.model = Model(inputs=input_layer, outputs=output_layer)
    optimizer = Adam(learning_rate=0.001)
    self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    # 3. 모델 학습 (오직 정상 데이터만으로)
    self.model.fit(
        X_train_normal,
        X_train_normal,
        epochs=self.config['epochs'],
        # batch_size=self.config['batch_size'],
        batch_size=64,
        shuffle=True,
        validation_split=0.2
    )

    # 4. 이상치 점수 계산 (테스트 데이터에 대한 재구성 오차)
    reconstructed_data = self.model.predict(self.data.X_test)
    reconstruction_error = np.mean(np.square(self.data.X_test - reconstructed_data), axis=1)

    # 5. 임계값 설정 및 최종 라벨 예측
    #    간단한 임계값 설정 방법: 훈련 데이터의 상위 5%를 임계값으로 설정
    train_normal_recon = self.model.predict(X_train_normal)
    train_normal_error = np.mean(np.square(X_train_normal - train_normal_recon), axis=1)
    threshold = np.percentile(train_normal_error, 97.5)

    # 예측된 이상치 점수와 임계값 비교하여 최종 라벨 결정
    y_pred = (reconstruction_error > threshold).astype(int)

    # 예측 결과와 하이퍼파라미터 저장
    self.data.y_pred = pd.Series(
        y_pred,
        index=self.data.name_test,
        name='label'
    )
    # self.models_hyperparameters = self.model.get_params()  # Keras 모델은 get_params()가 없으므로 다른 방식으로 저장 필요

    # return self.data.y_pred