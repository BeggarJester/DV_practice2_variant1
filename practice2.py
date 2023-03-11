import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

counter = 1

names_list1 = ['1 параметр', '2 параметр', '3 параметр', '4 параметр', '5 параметр', '6 параметр', '7 параметр',
               '8 параметр']

names_list2 = ['1 параметр', '2 параметр']

fluctuations = ['Fluctuations_X_in_the_left_bearing', 'Fluctuations_Y_in_the_left_bearing',
                'Fluctuations_X_in_the_right_bearing', 'Fluctuations_Y_in_the_right_bearing',
                'Fluctuations_X_in_the_left_bearing_smooth', 'Fluctuations_Y_in_the_left_bearing_smooth',
                'Fluctuations_X_in_the_right_bearing_smooth', 'Fluctuations_Y_in_the_right_bearing_smooth']

temperatures = ['Temperature_in_the_left_bearing', 'Temperature_in_the_right_bearing']

attributes = ['Fluctuations_X_in_the_left_bearing', 'Fluctuations_Y_in_the_left_bearing',
              'Fluctuations_X_in_the_right_bearing', 'Fluctuations_Y_in_the_right_bearing',
              'Fluctuations_X_in_the_left_bearing_smooth', 'Fluctuations_Y_in_the_left_bearing_smooth',
              'Fluctuations_X_in_the_right_bearing_smooth', 'Fluctuations_Y_in_the_right_bearing_smooth',
              'Temperature_in_the_left_bearing', 'Temperature_in_the_right_bearing']


def hist_for_2attributes_set(df_first, df_second, labels, ax, bin_num=25, y_scale='linear'):
    global counter
    counter = 1
    for i, col in enumerate(labels):
        ax[i].set_yscale(y_scale)
        ax[i].tick_params(labelsize=10)
        ax[i].set_xlabel(str(counter) + ' параметр')
        counter += 1

        # уберем лишние границы графика
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].grid()

        ax[i].hist(df_first[col], bin_num, density=False, color='#377eb8', alpha=.75)
        ax[i].hist(df_second[col], bin_num, density=False, color='#e41a1c', alpha=.45)


def hist_for_attributes_set(df, labels, ax, bin_num=25, y_scale='linear'):
    global counter
    counter = 1
    for i, col in enumerate(labels):
        ax[i].set_yscale(y_scale)
        ax[i].tick_params(labelsize=10)
        ax[i].set_xlabel(str(counter) + ' параметр')
        counter += 1

        # уберем лишние границы графика
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].grid()

        ax[i].hist(df[col], bin_num, density=False, color='#377eb8', alpha=.75)


def boxplot_for_attributes_set(df, labels, names, ax):
    props = dict(marker='o', markersize=3)

    for i, col in enumerate(labels):
        ax.boxplot(df[col], positions=[i + 1], flierprops=props)  # i + 1
        ax.tick_params(labelsize=10)
        ax.set_xticks(np.arange(len(labels)) + 1, names)


def draw_boxes(data_frame):
    fig, ax = plt.subplots(figsize=(17, 5))
    fig.set_figwidth(6)
    plt.xticks(rotation=90)
    boxplot_for_attributes_set(data_frame, fluctuations, names_list1, ax)
    ax.set_title('Ящик с усами: Показания колебаний', fontsize=16)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    np.arange(0, 101, 5)

    ax.grid(linestyle='--')
    fig.tight_layout(rect=[0, .03, 1, .95])
    plt.show()

    fig, ax = plt.subplots(figsize=(17, 5))
    fig.set_figwidth(6)
    plt.xticks(rotation=90)
    boxplot_for_attributes_set(data_frame, temperatures, names_list2, ax)
    ax.set_title('Ящик с усами: Показания температуры', fontsize=16)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    np.arange(0, 101, 5)

    ax.grid(linestyle='--')
    fig.tight_layout(rect=[0, .03, 1, .95])
    plt.show()


def draw_2histograms(data_frame1, data_frame2):
    fig, ax = plt.subplots(nrows=1, ncols=len(fluctuations), figsize=(20, 5))

    hist_for_2attributes_set(data_frame1, data_frame2, fluctuations, ax, bin_num=10, y_scale='log')

    ax[0].set_ylabel('Число записей')

    plt.suptitle('Гистограммы: Разница показаний колебаний', fontsize=16, y=.92)

    fig.tight_layout(rect=[0, .03, 1, .95])

    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=len(temperatures), figsize=(20, 5))

    hist_for_2attributes_set(data_frame1, data_frame2, temperatures, ax, bin_num=10, y_scale='log')

    ax[0].set_ylabel('Число записей')

    plt.suptitle('Гистограммы: Разница показаний температур', fontsize=16, y=.92)

    fig.tight_layout(rect=[0, .03, 1, .95])

    plt.show()


def draw_histograms(data_frame):
    fig, ax = plt.subplots(nrows=1, ncols=len(fluctuations), figsize=(20, 5))

    hist_for_attributes_set(data_frame, fluctuations, ax, bin_num=10, y_scale='log')

    ax[0].set_ylabel('Число записей')

    plt.suptitle('Гистограммы: Показания колебаний', fontsize=16, y=.92)

    fig.tight_layout(rect=[0, .03, 1, .95])

    plt.show()

    fig, ax = plt.subplots(nrows=1, ncols=len(temperatures), figsize=(20, 5))

    hist_for_attributes_set(data_frame, temperatures, ax, bin_num=10, y_scale='log')

    ax[0].set_ylabel('Число записей')

    plt.suptitle('Гистограммы: Показания температуры', fontsize=16, y=.92)

    fig.tight_layout(rect=[0, .03, 1, .95])

    plt.show()


def line_plot_for_attribute_set(df, labels, ax):
    for i, col in enumerate(labels):
        ax[i].set_xlabel(col)
        ax[i].plot(df[col], color='black', linewidth=.65)

        # уберем лишние границы графика
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].grid()
        ax[i].set_xlim(0, 1000)


def line_plot_for_2attribute_set(df_1, df_2, labels, ax):
    for i, col in enumerate(labels):
        ax[i].set_xlabel(col)
        ax[i].plot(df_1[col], color='black', linewidth=.65)
        ax[i].plot(df_2[col], color='red', linewidth=.65)

        # уберем лишние границы графика
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['top'].set_visible(False)
        ax[i].grid()
        ax[i].legend(['Без дефектов', 'С дефектом'])


def draw_line_plot(data_frame1, data_frame2):
    fig, ax = plt.subplots(nrows=len(fluctuations), ncols=1, figsize=(20, 25))

    line_plot_for_attribute_set(data_frame1, fluctuations, ax)
    fig.suptitle('Изменение показаний колебаний', fontsize=16, y=0.95)
    plt.show()

    fig, ax = plt.subplots(nrows=len(temperatures), ncols=1, figsize=(20, 25))

    line_plot_for_2attribute_set(data_frame1, data_frame2, temperatures, ax)
    fig.suptitle('Сравнение изменений показаний температур', fontsize=16, y=0.95)

    plt.show()


def heatmap(df, ax, title):
    corr = df.corr()

    # пропустим пары со слабой корреляцией, попробуем поэкспериментировать с этой настройкой
    df_corr = corr[(corr >= .5) | (corr <= -.5)]

    # print(f'Empty columns: {get_empty_columns_names(df_corr)}')

    mask = np.triu(np.ones_like(corr))
    sns.heatmap(
        df_corr,
        cbar=False,
        mask=mask,
        ax=ax,
        annot=True,
        cmap='coolwarm',
        vmin=-1,
        vmax=1,
    )

    ax.title.set_text(title)
    ax.tick_params(
        top=False,
        bottom=False,
        left=False,
        right=False,
    )


def draw_heatmap(data_frame):
    fig, ax = plt.subplots(figsize=(15, 15), constrained_layout=True)
    heatmap(data_frame[attributes], ax, 'Матрица корреляции')
    plt.show()


def get_pca(data_frame):
    x = data_frame[attributes].values

    # нормализуем значения
    x = StandardScaler().fit_transform(x)

    pca = PCA(n_components=3)
    pca.fit_transform(x)
    print(f'Explained variance: {pca.explained_variance_ratio_}\tSum: {pca.explained_variance_ratio_.sum()}\n')


if __name__ == '__main__':

    # Загружаем данные из файлов
    df1 = pd.read_csv('datas/data_5_1.csv')
    df2 = pd.read_csv('datas/data_5_2.csv')
    df3 = pd.read_csv('datas/data_5_3.csv')
    df4 = pd.read_csv('datas/data_5_4.csv')

    # Устанавливаем настройки для полного вывода статистики о файле
    pd.set_option('display.expand_frame_repr', False)

    # Выводим информацию о файле
    for frame in [df1, df2, df3, df4]:
        print(frame.info())

    # Выводим статистику о файле
    for frame in [df1, df2, df3, df4]:
        print(frame.describe())

    # Устанавливаем стиль графиков
    plt.style.use('bmh')

    # Для каждого набора данных рисуем гистограммы и графики с усами
    for frame in [df1, df2, df3, df4]:
        draw_histograms(frame)
        draw_boxes(frame)

    # Для каждого набора данных с дефектами рисуем сравнительные гистограммы с набором без дефектов
    for frame in [df2, df3, df4]:
        draw_2histograms(df1, frame)

    # Для каждого набора данных рисуем распределение значений по времени
    for frame in [df2, df3, df4]:
        draw_line_plot(df1, frame)

    # Для каждого набора данных рисуем матрицу корреляции
    for frame in [df1, df2, df3, df4]:
        draw_heatmap(frame)

    # Для каждого набора данных применим преобразование PCA
    for frame in [df1, df2, df3, df4]:
        get_pca(frame)
