from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def save_PCA_LDA(self, graph_name):
    if graph_name == 'after_split':
        df_train = self.df_train.copy()
        df_test = self.df_test.copy()
    else:
        df_train = self.df_train_after_sampling.copy()
        df_test = self.df_test.copy()

    df_train['set'] = 'train'
    df_test['set'] = 'test'

    # 훈련 및 테스트 데이터프레임 결합
    combined_df = pd.concat([df_train, df_test], ignore_index=True)

    # set, label 컬럼 분리
    plot_info = combined_df[['set', 'label']].copy()
    features = combined_df.drop(columns=['set', 'label'])

    # NaN 값 0으로 채우기
    features = features.fillna(0)

    # PCA 적용 (3차원으로 축소)
    print("PCA 적용하여 3차원으로 축소 중...")
    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(features)

    # 시각화를 위한 데이터프레임
    pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])
    final_plot_df = pd.concat([pca_df, plot_info], axis=1)

    # 3D 시각화
    print("3D 그래프 생성 중...")
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')

    final_plot_df['group'] = final_plot_df['set'] + '_' + final_plot_df['label'].astype(str)
    colors = {
        'train_False': 'lightgray',
        'train_True': 'deepskyblue',
        'test_False': 'dimgray',
        'test_True': 'red'
    }

    for index, row in final_plot_df.iterrows():
        ax.scatter(
            row['PC1'], row['PC2'], row['PC3'],
            c=colors[row['group']],
            marker='o',
            s=50,
            alpha=0.6
        )

    # 범위 설정 및 극단값 제외
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

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Train, False', markerfacecolor=colors['train_False'],
               markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Train, True', markerfacecolor=colors['train_True'],
               markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Test, False', markerfacecolor=colors['test_False'],
               markersize=10),
        Line2D([0], [0], marker='o', color='w', label='Test, True', markerfacecolor=colors['test_True'],
               markersize=10)
    ]
    ax.legend(handles=legend_elements, title='Legend')

    # 그래프를 파일로 저장
    plt.savefig(self.config['result_folder_path'] + '/data_distribution_pca_' + graph_name + '.png')
    plt.show()
    print("데이터 분포 시각화 그래프가 'data_distribution_pca.png' 파일로 저장되었습니다.")

    true_only_df = final_plot_df[final_plot_df['label'] == True].copy()
    fig_true = plt.figure(figsize=(15, 12))
    ax_true = fig_true.add_subplot(111, projection='3d')

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

    legend_elements_true = [
        Line2D([0], [0], marker='o', color='w', label='Train, True', markerfacecolor=colors_true['train'],
               markersize=10),
        Line2D([0], [0], marker='X', color='w', label='Test, True', markerfacecolor=colors_true['test'],
               markersize=10)
    ]
    ax_true.legend(handles=legend_elements_true, title='Legend (True Labels)')

    plt.savefig(self.config['result_folder_path'] + '/data_distribution_pca_test_only_' + graph_name + '.png')
    plt.show()
    print("테스트 데이터 전용 시각화 그래프가 'data_distribution_pca_test_only.png' 파일로 저장되었습니다.")

    print("LDA 적용하여 1차원으로 축소 중...")
    lda = LDA(n_components=1)

    lda.fit(self.df_x_train_flatten,  list(self.df_y_train['label']))

    lda_train = lda.transform(self.df_x_train_flatten)
    lda_test = lda.transform(self.df_x_test_flatten)

    # LDA 시각화를 위한 데이터프레임
    df_train_lda = pd.DataFrame(data=lda_train, columns=['LD1'])
    df_train_lda['set'] = 'train'
    df_train_lda['label'] = list(self.df_y_train['label'])

    df_test_lda = pd.DataFrame(data=lda_test, columns=['LD1'])
    df_test_lda['set'] = 'test'
    df_test_lda['label'] = list(self.df_y_test['label'])

    final_plot_df_lda = pd.concat([df_train_lda, df_test_lda], ignore_index=True)

    # 1D LDA 시각화 (스트립 플롯)
    print("1D 스트립 플롯 생성 중...")
    plt.figure(figsize=(15, 10))

    final_plot_df_lda['group'] = final_plot_df_lda['set'] + '_' + final_plot_df_lda['label'].astype(str)
    colors_lda = {
        'train_False': 'lightgray',
        'train_True': 'deepskyblue',
        'test_False': 'dimgray',
        'test_True': 'red'
    }

    sns.stripplot(
        x='LD1', y='group', data=final_plot_df_lda,
        hue='group', palette=colors_lda, jitter=0.3, size=6, alpha=0.8,
        order=['train_False', 'train_True', 'test_False', 'test_True'],
        legend=False
    )

    plt.title('1D LDA of Train vs Test Data Distribution with Labels', fontsize=16)
    plt.xlabel('Linear Discriminant 1 (LD1)', fontsize=12)
    plt.ylabel('Group', fontsize=12)
    plt.grid(axis='x')

    legend_elements_lda = [
        Line2D([0], [0], marker='o', color='w', label='Train, False',
               markerfacecolor=colors_lda['train_False'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Train, True',
               markerfacecolor=colors_lda['train_True'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Test, False',
               markerfacecolor=colors_lda['test_False'], markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Test, True',
               markerfacecolor=colors_lda['test_True'], markersize=8)
    ]

    plt.legend(handles=legend_elements_lda, title='Group')

    plt.savefig(self.config['result_folder_path'] + '/data_distribution_lda_' + graph_name + '.png')
    plt.show()
    print("LDA 시각화 그래프가 'data_distribution_lda.png' 파일로 저장되었습니다.")
