from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def save_PCA_LDA(self, graph_name):
    x_train = self.df_x_train.drop(columns=["company_id"]).copy()
    x_test = self.df_x_test.drop(columns=["company_id"]).copy()
    x_train['set'] = 'train'
    x_test['set'] = 'test'
    x_train['label'] = self.df_y_train
    x_test['label'] = self.df_y_test
    combined_df = pd.concat([x_train, x_test], ignore_index=True)
    plot_info = combined_df[['set', 'label']]
    features = combined_df.drop(columns=['set', 'label'])
    features = features.fillna(0)

    # 3. PCA 적용 (3차원으로 축소)
    print("PCA 적용하여 3차원으로 축소 중...")
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features)

    # 4. 시각화를 위한 최종 데이터프레임 생성
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    final_plot_df = pd.concat([pca_df, plot_info], axis=1)

    # 5. 3D 시각화
    print("3D 그래프 생성 중...")
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    final_plot_df['group'] = final_plot_df['set'] + '_' + final_plot_df['label'].astype(str)
    colors = {
        'train_False': 'lightgray',
        'train_True': 'deepskyblue',  # 훈련 데이터의 True
        'test_False': 'dimgray',
        'test_True': 'red'  # 테스트 데이터의 True (가장 중요)
    }

    # [수정 2] 새로운 'group' 컬럼을 기준으로 색상을 지정하고, 마커는 'o'로 통일
    for index, row in final_plot_df.iterrows():
        ax.scatter(
            row['PC1'], row['PC2'], row['PC3'],
            c=colors[row['group']],  # 그룹에 맞는 색상 사용
            marker='o',  # 모든 마커를 원으로 통일
            s=50,
            alpha=0.6
        )

    # ax.set_xlim([-7, 5])
    # ax.set_ylim([-10, 40])
    # ax.set_zlim([-7, 5])
    self.x_range = [final_plot_df['PC1'].quantile(0.01), final_plot_df['PC1'].quantile(0.99)]
    self.y_range = [final_plot_df['PC2'].quantile(0.01), final_plot_df['PC2'].quantile(0.99)]
    self.z_range = [final_plot_df['PC3'].quantile(0.01), final_plot_df['PC3'].quantile(0.99)]

    ax.set_xlim(self.x_range[0], self.x_range[1])
    ax.set_ylim(self.y_range[0], self.y_range[1])
    ax.set_zlim(self.z_range[0], self.z_range[1])

    ax.set_title('3D PCA of Train vs Test Data Distribution with Labels', fontsize=16)
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_zlabel('Principal Component 3', fontsize=12)

    # 범례(Legend) 수동 생성
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Train, False', markerfacecolor=colors['train_False'],
               markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Train, True', markerfacecolor=colors['train_True'],
               markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Test, False', markerfacecolor=colors['test_False'],
               markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Test, True', markerfacecolor=colors['test_True'], markersize=10)
    ]
    ax.legend(handles=legend_elements, title='Legend')

    # 그래프를 파일로 저장
    plt.savefig(self.config['result_folder_path'] + '/data_distribution_pca_' + graph_name + '.png')
    plt.show()
    print("데이터 분포 시각화 그래프가 'data_distribution_pca.png' 파일로 저장되었습니다.")

    true_only_df = final_plot_df[final_plot_df['label'] == True].copy()
    fig_true = plt.figure(figsize=(15, 12))
    ax_true = fig_true.add_subplot(111, projection='3d')

    # 'train'과 'test'를 구분할 새로운 색상 맵을 정의
    colors_true = {'train': 'deepskyblue', 'test': 'red'}
    markers_true = {'train': 'o', 'test': 'X'}

    # 필터링된 'True' 데이터만 플로팅
    for index, row in true_only_df.iterrows():
        ax_true.scatter(
            row['PC1'], row['PC2'], row['PC3'],
            c=colors_true[row['set']],
            marker=markers_true[row['set']],
            s=80, alpha=0.8
        )

    ax_true.set_xlim(self.x_range[0], self.x_range[1])
    ax_true.set_ylim(self.y_range[0], self.y_range[1])
    ax_true.set_zlim(self.z_range[0], self.z_range[1])

    ax_true.set_title('3D PCA of True-Labeled Data Only (Train vs Test)', fontsize=16)
    ax_true.set_xlabel('Principal Component 1', fontsize=12)
    ax_true.set_ylabel('Principal Component 2', fontsize=12)
    ax_true.set_zlabel('Principal Component 3', fontsize=12)

    # 범례를 'Train, True'와 'Test, True'만 표시하도록 단순화
    legend_elements_true = [
        Line2D([0], [0], marker='o', color='w', label='Train, True', markerfacecolor=colors_true['train'],
               markersize=10),
        Line2D([0], [0], marker='X', color='w', label='Test, True', markerfacecolor=colors_true['test'], markersize=10)
    ]
    ax_true.legend(handles=legend_elements_true, title='Legend (True Labels)')

    plt.savefig(self.config['result_folder_path'] + '/data_distribution_pca_test_only_' + graph_name + '.png')
    plt.show()
    print("테스트 데이터 전용 시각화 그래프가 'data_distribution_pca_test_only.png' 파일로 저장되었습니다.")

    x_train_scaled = self.df_x_train.drop(columns=["company_id"]).copy().fillna(0)
    x_test_scaled = self.df_x_test.drop(columns=["company_id"]).copy().fillna(0)
    y_train = self.df_y_train
    y_test = self.df_y_test

    # 3. LDA 적용
    print("LDA 적용하여 1차원으로 축소 중...")
    lda = LDA(n_components=1)

    # 중요: 훈련 데이터로만 LDA 모델을 학습(fit)합니다.
    lda.fit(x_train_scaled, y_train)

    # 학습된 LDA로 훈련 데이터와 테스트 데이터를 모두 변환(transform)합니다.
    lda_train = lda.transform(x_train_scaled)
    lda_test = lda.transform(x_test_scaled)

    # 4. 시각화를 위한 최종 데이터프레임 생성
    df_train_lda = pd.DataFrame(data=lda_train, columns=['LD1'])
    df_train_lda['set'] = 'train'
    df_train_lda['label'] = y_train.values

    df_test_lda = pd.DataFrame(data=lda_test, columns=['LD1'])
    df_test_lda['set'] = 'test'
    df_test_lda['label'] = y_test.values

    final_plot_df = pd.concat([df_train_lda, df_test_lda], ignore_index=True)

    # 5. 1D 시각화 (스트립 플롯)
    print("1D 스트립 플롯 생성 중...")
    plt.figure(figsize=(15, 10))

    final_plot_df['group'] = final_plot_df['set'] + '_' + final_plot_df['label'].astype(str)
    colors = {
        'train_False': 'lightgray',
        'train_True': 'deepskyblue',
        'test_False': 'dimgray',
        'test_True': 'red'
    }

    sns.stripplot(
        x='LD1', y='group', data=final_plot_df,
        hue='group', palette=colors, jitter=0.3, size=6, alpha=0.7,
        order=['train_False', 'train_True', 'test_False', 'test_True'],
        legend=False
    )

    plt.title('1D LDA of Train vs Test Data Distribution with Labels', fontsize=16)
    plt.xlabel('Linear Discriminant 1 (LD1)', fontsize=12)
    plt.ylabel('Group', fontsize=12)
    plt.grid(axis='x')
    from matplotlib.lines import Line2D
    # PCA 플롯에서 사용한 것과 동일한 방식으로, 각 그룹에 대한 범례 요소를 만듭니다.
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Train, False',
               markerfacecolor=colors['train_False'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Train, True',
               markerfacecolor=colors['train_True'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Test, False',
               markerfacecolor=colors['test_False'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Test, True',
               markerfacecolor=colors['test_True'], markersize=8)
    ]

    # 수동으로 생성한 요소들을 사용하여 범례를 만듭니다.
    plt.legend(handles=legend_elements, title='Group')

    plt.savefig(self.config['result_folder_path'] + '/data_distribution_lda_' + graph_name + '.png')
    plt.show()
    print("LDA 시각화 그래프가 'data_distribution_lda.png' 파일로 저장되었습니다.")
