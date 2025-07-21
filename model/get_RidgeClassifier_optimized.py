import pandas as pd
import joblib
import numpy as np
import optuna
from sklearn.linear_model import RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, accuracy_score, fbeta_score

def objective(self, trial):
    # RidgeClassifer의 class_weight 최적화
    class_weight_False = trial.suggest_int("class_weight_False", 1, 2000)
    class_weight_True = trial.suggest_int("class_weight_True", 1, 2000)
    class_weight = {0: class_weight_False, 1: class_weight_True}

    model = RidgeClassifier(
        random_state=self.config['random_state'],
        class_weight=class_weight
    )

    # company_id 제외하고 X 데이터 준비
    # errors='ignore'는 company_id가 없을 경우 에러를 무시합니다.
    X = self.data.X_train
    y = self.data.y_train

    # 교차 검증 설정 (StratifiedKFold는 불균형 데이터에 적합)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.get('random_state', 42))

    # 평가지표 정의: "경영악화" 클래스(레이블 1로 가정)에 대한 recall과 accuracy, F2-score
    # recall_score의 pos_label=1: 레이블 1을 positive 클래스로 간주 (경영악화)
    # zero_division=0: recall 계산 시 분모가 0이 되는 경우 0으로 처리 (경고 방지)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'recall_positive': make_scorer(recall_score, pos_label=1, zero_division=0),
        'f2_positive': make_scorer(fbeta_score, beta=2, pos_label=1, zero_division=0)  # Recall에 더 큰 가중치
    }

    try:
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        mean_accuracy = np.mean(scores['test_accuracy'])
        mean_recall_positive = np.mean(scores['test_recall_positive'])
        mean_f2_positive = np.mean(scores['test_f2_positive'])

        # mean_train_accuracy = np.mean(scores['train_accuracy'])
        # Optuna trial에 사용자 속성으로 다른 지표들 저장 (나중에 확인용)
        trial.set_user_attr("mean_accuracy", mean_accuracy)
        trial.set_user_attr("mean_recall_positive", mean_recall_positive)

        # Optuna는 이 반환값을 최대화하려고 시도합니다. F2-score를 사용하여 recall에 중점.
        return mean_recall_positive

    except ValueError as e:
        # CV 중 특정 폴드에 positive 클래스가 없는 등의 예외 처리
        print(f"Trial {trial.number} - 교차 검증 중 오류 발생: {e}. 낮은 점수 반환.")
        return -1.0  # 실패한 trial에 대해 매우 낮은 점수 반환


def get_RidgeClassifier_optimized(self):
    print("\n=== RidgeClassifier 하이퍼파라미터 최적화 시작 ===")

    # Optuna study 생성: F2-score를 최대화하는 방향으로 탐색
    # study_name을 지정하면 나중에 재개하거나 여러 워커에서 공유 가능
    study = optuna.create_study(direction='maximize', study_name='ridge_recall_accuracy_opt')

    # 최적화 실행 (n_trials는 시도 횟수, 필요에 따라 늘리거나 줄임)
    # timeout 파라미터로 시간 제한도 가능: study.optimize(lambda trial: self.objective(trial), n_trials=50, timeout=3600)
    study.optimize(lambda trial: objective(self, trial), n_trials=20)  # objective 메서드가 self를 사용하므로 람다로 감싸줌

    print("\n최적화 결과:")
    best_trial = study.best_trial
    print(f"  최적 F2-score (목표): {best_trial.value:.4f}")
    print("  최적 하이퍼파라미터:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # 저장된 사용자 속성(다른 평가지표) 출력
    print("  해당 trial의 다른 평균 평가지표들:")
    if best_trial.user_attrs:
        for key, value in best_trial.user_attrs.items():
            print(f"    {key}: {value:.4f}")

    # 최적 하이퍼파라미터 추출
    best_params = best_trial.params
    class_weight = {0: best_params['class_weight_False'], 1: best_params['class_weight_True']}


    # 최적 파라미터로 최종 모델 생성

    self.model = RidgeClassifier(

        random_state=self.config.get('random_state', 42),
        class_weight=class_weight
    )

    print("\n최적 파라미터로 전체 훈련 데이터에 모델 재학습 중...")
    # 전체 훈련 데이터로 최종 모델 학습
    X_train_final = self.data.X_train
    y_train_final = self.data.y_train
    self.model.fit(X_train_final, y_train_final)
    print("모델 재학습 완료.")

    # 테스트 데이터로 예측 (기존 코드와 동일)
    X_test_final = self.data.X_test
    self.data.y_pred = pd.Series(
        self.model.predict(X_test_final),
        index=self.data.name_test,
        name='label'
    )

    self.models_hyperparameters = self.model.get_params()

    if self.config.get('save_model', False):
        folder_path = self.config.get('result_folder_path', '.')
        # import os; os.makedirs(folder_path, exist_ok=True) # 폴더가 없다면 생성
        model_path = f"{folder_path}/RidgeClassifier_optimized_model.joblib"
        joblib.dump(self.model, model_path)
        print(f"최적화된 모델 저장 완료: {model_path}")

    print("=== RidgeClassifier 하이퍼파라미터 최적화 완료 ===")
    return self.model  # 필요시 학습된 모델 반환