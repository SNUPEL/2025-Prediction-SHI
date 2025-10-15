import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.svm import SVC
from sklearn.svm import NuSVC
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, recall_score, accuracy_score, fbeta_score


def objective(self, trial):
    if self.config['model_type'] == 'SVC':      # 선택된 모델이 SVC일 경우 실행
        # SVC 하이퍼파라미터 탐색 공간 정의
        C = trial.suggest_float('C', 0.1, 0.2)      # 규제 파라미터, 0.1~0.2 사이의 실수를 탐색

        # SVC의 class_weight 최적화
        # class_weight_False = trial.suggest_int("class_weight_False", 1, 10)
        class_weight_False = 1      # '거래중' 클래스에 부여하는 가중치, 1로 고정
        class_weight_True = trial.suggest_int("class_weight_True", 700, 800)        # '경영악화' 클래스에 부여하는 가중치, 700~800 사이의 정수를 탐색
        class_weight = {0: class_weight_False, 1: class_weight_True}

        model = SVC(        # trial을 통해 선택된 C, class_weight를 기반으로 SVC 실행
            C=C,
            class_weight=class_weight,
            random_state=self.config.get('random_state', 42),  # config에 없으면 기본값 42 사용
        )
    elif self.config['model_type'] == 'nuSVC':      # 선택된 모델이 nuSVC일 경우 실행
        class_weight_False = trial.suggest_int("class_weight_False", 1, 10)     # '거래중' 클래스에 부여하는 가중치, 1~10 사이의 정수를 탐색
        # class_weight_False = 1
        class_weight_True = trial.suggest_int("class_weight_True", 700, 800)    # '경영악화' 클래스에 부여하는 가중치, 700~800 사이의 정수를 탐색
        class_weight = {0: class_weight_False, 1: class_weight_True}
        model = NuSVC(      # trial을 통해 선택된 class_weight를 기반으로 nuSVC 실행
            nu=0.01,
            class_weight=class_weight,
            random_state=self.config.get('random_state', 42)
        )

    # company_id 제외하고 X 데이터 준비
    # errors='ignore'는 company_id가 없을 경우 에러를 무시합니다.
    X = self.data.df_x_train_flatten
    y = self.data.df_y_train

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
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)     # 교차 검증 실행->각 변수에 대해 점수화

        mean_accuracy = np.mean(scores['test_accuracy'])        # 전체 trial의 평균 accuracy
        mean_recall_positive = np.mean(scores['test_recall_positive'])      # 전체 trial의 평균 recall
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


# get_SVC 메서드를 최적화 기능이 추가된 새 메서드로 대체하거나 새로 만듭니다.
def get_SVC_optimized(self):  # 함수 이름을 변경하거나 기존 함수를 수정
    print("\n=== SVC 하이퍼파라미터 최적화 시작 ===")

    # Optuna study 생성: F2-score를 최대화하는 방향으로 탐색
    # study_name을 지정하면 나중에 재개하거나 여러 워커에서 공유 가능
    study = optuna.create_study(direction='maximize', study_name='svc_recall_accuracy_opt')

    # 최적화 실행 (n_trials는 시도 횟수, 필요에 따라 늘리거나 줄임)
    # timeout 파라미터로 시간 제한도 가능: study.optimize(lambda trial: self.objective(trial), n_trials=50, timeout=3600)
    study.optimize(lambda trial: objective(self, trial), n_trials=20)  # objective 메서드가 self를 사용하므로 람다로 감싸줌

    print("\n최적화 결과:")
    best_trial = study.best_trial
    print(f"  최적 F2-score (목표): {best_trial.value:.4f}")
    print("  최적 하이퍼파라미터:")      # C, class_weight_True의 값 출력
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    # 저장된 사용자 속성(다른 평가지표) 출력
    print("  해당 trial의 다른 평균 평가지표들:")
    if best_trial.user_attrs:
        for key, value in best_trial.user_attrs.items():
            print(f"    {key}: {value:.4f}")

    # 최적 하이퍼파라미터 추출
    best_params = best_trial.params
    C = best_params['C']
    class_weight = {0: 1, 1: best_params['class_weight_True']}

    # 최적 파라미터로 최종 모델 생성
    self.model = SVC(
        C=C,
        class_weight=class_weight,
        random_state=self.config.get('random_state', 42)
    )

    print("\n최적 파라미터로 전체 훈련 데이터에 모델 재학습 중...")
    # 전체 훈련 데이터로 최종 모델 학습
    df_x_train_flatten = self.data.df_x_train_flatten
    df_y_train = self.data.df_y_train
    self.model.fit(df_x_train_flatten, df_y_train['label'])
    print("모델 재학습 완료.")

    # 테스트 데이터로 예측 (기존 코드와 동일)
    X_test_final = self.data.df_x_test_flatten
    self.data.df_y_pred = pd.DataFrame(
        self.model.predict(X_test_final),
        index=self.data.name_test,
        columns=['label']
    )

    self.models_hyperparameters = self.model.get_params()

    if self.config.get('save_model', False):        # 학습된 모델 저장
        folder_path = self.config.get('result_folder_path', '.')
        # import os; os.makedirs(folder_path, exist_ok=True) # 폴더가 없다면 생성
        model_path = f"{folder_path}/AdaBoostClassifier_optimized_model.joblib"
        joblib.dump(self.model, model_path)
        print(f"최적화된 모델 저장 완료: {model_path}")

    print("=== SVC 하이퍼파라미터 최적화 완료 ===")
    return self.model  # 필요시 학습된 모델 반환


# get_nuSVC 메서드를 최적화 기능이 추가된 새 메서드로 대체하거나 새로 만듭니다.
def get_nuSVC_optimized(self):  # 함수 이름을 변경하거나 기존 함수를 수정
    print("\n=== SVC 하이퍼파라미터 최적화 시작 ===")

    # Optuna study 생성: F2-score를 최대화하는 방향으로 탐색
    # study_name을 지정하면 나중에 재개하거나 여러 워커에서 공유 가능
    study = optuna.create_study(direction='maximize', study_name='nusvc_recall_accuracy_opt')

    # 최적화 실행 (n_trials는 시도 횟수, 필요에 따라 늘리거나 줄임)
    # timeout 파라미터로 시간 제한도 가능: study.optimize(lambda trial: self.objective(trial), n_trials=50, timeout=3600)
    study.optimize(lambda trial: objective(self, trial), n_trials=20)  # objective 메서드가 self를 사용하므로 람다로 감싸줌

    print("\n최적화 결과:")
    best_trial = study.best_trial
    print(f"  최적 F2-score (목표): {best_trial.value:.4f}")
    print("  최적 하이퍼파라미터:")      # class_weight_False, class_weight_True 값 출력
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
    self.model = NuSVC(
        nu=0.01,
        class_weight=class_weight,
        random_state=self.config.get('random_state', 42)
    )

    print("\n최적 파라미터로 전체 훈련 데이터에 모델 재학습 중...")
    # 전체 훈련 데이터로 최종 모델 학습
    df_x_train_flatten = self.data.df_x_train_flatten
    df_y_train = self.data.df_y_train
    self.model.fit(df_x_train_flatten, df_y_train['label'])
    print("모델 재학습 완료.")

    # 테스트 데이터로 예측 (기존 코드와 동일)
    X_test_final = self.data.df_x_test_flatten
    self.data.df_y_pred = pd.DataFrame(
        self.model.predict(X_test_final),
        index=self.data.name_test,
        columns=['label']
    )

    self.models_hyperparameters = self.model.get_params()

    if self.config.get('save_model', False):        # 학습된 모델 저장
        folder_path = self.config.get('result_folder_path', '.')
        # import os; os.makedirs(folder_path, exist_ok=True) # 폴더가 없다면 생성
        model_path = f"{folder_path}/NuSVC_optimized_model.joblib"
        joblib.dump(self.model, model_path)
        print(f"최적화된 모델 저장 완료: {model_path}")

    print("=== nuSVC 하이퍼파라미터 최적화 완료 ===")
    return self.model  # 필요시 학습된 모델 반환


# --- 사용 예시 (클래스 외부에서 호출한다고 가정) ---
# trainer = MyModelTrainer(data_object, config_dict) # 실제 객체들로 초기화
# optimized_adaboost_model = trainer.get_AdaBoostClassifier_optimized()

# # 이후 optimized_adaboost_model 또는 trainer.data.df_y_pred를 사용하여 성능 평가
# from sklearn.metrics import classification_report, confusion_matrix
# print("\n최적화된 모델 테스트 세트 성능:")
# print(classification_report(trainer.data.df_y_test, trainer.data.df_y_pred, target_names=['거래중', '경영악화'])) # 클래스 이름은 실제에 맞게 조정
# print(confusion_matrix(trainer.data.df_y_test, trainer.data.df_y_pred))
