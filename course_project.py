from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io as sio
import scipy.stats as stats
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import cv2

from PIL import Image
import time
import argparse

import pyximport
pyximport.install()
import pyflow


# спросить про то какой коэффициент использовать?

############################
# Загрузка данных
############################


def load_data(path):
    """
    Функция загружающая данные из .mat файла
    :param path: путь к файлу
    :return:
    dt: промежуток между измерениями
    t_start: начало измерений
    t_end: конец измерений
    data_values: развёрнутые данные (16, 16, число измерений)
    rows_num, col_num: (16, 16)
    """
    mat = sio.loadmat(path)

    sign_bb = np.array(mat['sign_bb'])
    data = np.array(mat['Data'])

    dt = data[0][1][0][0] * 1.0e-3 # разница между измерениями
    t_start = data[1][1][0][0]
    t_end = t_start + sign_bb.shape[2] * dt # время начала + число измерений * dt
    print(dt, t_start, t_end)
    data_values = np.zeros(sign_bb.shape)

    for i in range(sign_bb.shape[2]):
        data_values[:, :, i] = np.rot90(sign_bb[:, :, i], 2)
    rows_num = sign_bb.shape[0]
    col_num = sign_bb.shape[1]
    return dt, t_start, t_end, data_values, rows_num, col_num


############################
# Корреляции между столбцами
# Разделение по экватору
############################


def corr_between_col_sums(data_values, left_idx, right_idx):
    """
    Корреляция между суммами столбцов
    Плохо: получается всегда близка к 1
    КОЭФФИЦИЕНТ СПИРМЕНА
    :param data_values: (16, 16, ...)
    :param left_idx: временной индекс начала
    :param right_idx: временной индекс конца
    :return:
    """
    upper_values = data_values[16 // 2:, :, left_idx:right_idx]
    lower_values = data_values[:16 // 2, :, left_idx:right_idx]
    upper_sum = upper_values.sum(axis=0)
    lower_sum = lower_values.sum(axis=0)
    res = np.zeros(lower_sum.shape[1])
    for i in range(lower_sum.shape[1]):
        res[i] = stats.spearmanr(upper_sum[:, i], lower_sum[:, i])[0]
    return res


def col_sums_plot(data_values, left_idx, right_idx):
    # Построение графика корреляции сумм столбцов
    corr_col_sums = corr_between_col_sums(data_values, left_idx, right_idx)
    fig, ax = plt.subplots()
    ax.plot(np.linspace(135, 175, left_idx - right_idx), corr_col_sums,
            lw=0.5)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel('timeline, ms')
    ax.set_ylabel('Spearman corr. coeff.')
    plt.title('Correlation between sums of columns')
    plt.show()
    fig.savefig("./col_sums_corr.png", format="png")


def col_col_corr(column_slice_vals, left_idx, right_idx):
    """
    Корреляция между суммами столбец-столбец
    Разделение по экватору
    КОЭФФИЦИЕНТ СПИРМЕНА
    :param column_slice_vals:
    :param left_idx:
    :param right_idx:
    :return:
    """
    upper_values = column_slice_vals[8:, left_idx:right_idx]
    lower_values = column_slice_vals[:8, left_idx:right_idx]
    upper_sum = upper_values.sum(axis=0)
    lower_sum = lower_values.sum(axis=0)
    res = stats.spearmanr(upper_sum, lower_sum)[0]
    return res


def col_correlation_plot(data_values, left_idx, right_idx, dt):
    # Построение графиков корреляции столбцов во временнном окне 1 мс
    fig, ax = plt.subplots(4, 4, figsize=(16, 16), dpi=160)
    for column in range(16):
        print(column)
        corr_col_col = [col_col_corr(data_values[:, column, :], i, i + int(1 // dt))
                        for i in range(left_idx, right_idx - int(1 // dt))]
        cur_ax = ax[column // 4, column % 4]
        cur_ax.plot(np.linspace(135, 174,
                                right_idx - int(1 // dt) - left_idx), corr_col_col)
        cur_ax.set_ylim(-1.0, 1.0)
        cur_ax.set_xlabel('timeline, ms')
    fig.suptitle('Correlation between columns within the window of 1 ms')
    plt.show()
    fig.savefig("./col_col_corr.png", format="png")


############################
# Корреляции между строками
# Разделение по: 0-3, 4-15
############################


def draw_sum_gr(data_values, left_idx, right_idx):
    """
    Графики сумм внутренней и внешней частей
    :param data_values:
    :param left_idx:
    :param right_idx:
    :return:
    """
    inner_values = data_values[:, :4, left_idx:right_idx]
    outer_values = data_values[:, 4:8, left_idx:right_idx]
    inner_sum = inner_values.sum(axis=(0, 1))
    outer_sum = outer_values.sum(axis=(0, 1))
    fig, ax = plt.subplots()
    ax.plot(np.linspace(135, 175, right_idx - left_idx), inner_sum, label='Inner')
    ax.plot(np.linspace(135, 175, right_idx - left_idx), outer_sum, label='Outer')
    ax.set_xlabel('timeline, ms')
    plt.legend()
    plt.show()
    fig.savefig("./inner_outer_sums_inner.png", format="png")


def draw_sum_gr_normed(data_values, left_idx, right_idx):
    """
    Графики сумм внутренней и внешней частей
    Нормировано по числу столбцов
    :param data_values:
    :param left_idx:
    :param right_idx:
    :return:
    """
    inner_values = data_values[:, :4, left_idx:right_idx]
    outer_values = data_values[:, 4:, left_idx:right_idx]
    inner_sum = inner_values.sum(axis=(0, 1)) / 4
    outer_sum = outer_values.sum(axis=(0, 1)) / 12
    fig, ax = plt.subplots()
    ax.plot(np.linspace(135, 175, right_idx - left_idx), inner_sum, label='Inner')
    ax.plot(np.linspace(135, 175, right_idx - left_idx), outer_sum, label='Outer')
    ax.set_xlabel('timeline, ms')
    plt.legend()
    plt.show()
    fig.savefig("./normed_inner_outer_sums.png", format="png")


def row_to_row_corr(row_slice_vals, left_idx, right_idx):
    """
    Корреляция между суммами строка-строка
    :param column_slice_vals:
    :param left_idx:
    :param right_idx:
    :return:
    """
    inner_values = row_slice_vals[:4, left_idx:right_idx]
    outer_values = row_slice_vals[4:8, left_idx:right_idx]
    inner_sum = inner_values.sum(axis=0)
    outer_sum = outer_values.sum(axis=0)
    res = stats.spearmanr(inner_sum, outer_sum)[0]
    return res


def row_correlation(data_values, left_idx, right_idx, dt):
    """
    16 графиков корреляции столбец-столбец с временнным окном 1 мс
    :param data_values:
    :param left_idx:
    :param right_idx:
    :param dt:
    :return:
    """
    fig, ax = plt.subplots(4, 4, figsize=(16, 16), dpi=160)
    for row in range(16):
        print(row)
        corr_row_row = [row_to_row_corr(data_values[row, :, :], i, i + int(1 // dt))
                        for i in range(left_idx, right_idx - int(1 // dt))]
        cur_ax = ax[row // 4, row % 4]
        cur_ax.plot(np.linspace(135, 174,
                                right_idx - int(1 // dt) - left_idx), corr_row_row)
        cur_ax.set_ylim(-1.0, 1.0)
        cur_ax.set_xlabel('timeline, ms')
    fig.suptitle('Correlation between rows within the window of 1 ms')
    plt.show()
    fig.savefig("./row_row_corr_inner.png", format="png")


def sums_corr(data_values, left_idx, right_idx):
    inner_values = data_values[:, :8, left_idx:right_idx]
    outer_values = data_values[:, 8:, left_idx:right_idx]
    inner_sum = inner_values.sum(axis=(0, 1))
    outer_sum = outer_values.sum(axis=(0, 1))
    res = stats.spearmanr(inner_sum, outer_sum)[0]
    return res


def in_out_sums_corr(data_values, left_idx, right_idx, dt):
    fig, ax = plt.subplots()
    corr_row_row = [sums_corr(data_values, i, i + int(1 // dt))
                    for i in range(left_idx, right_idx - int(1 // dt))]
    ax.plot(np.linspace(135, 174, right_idx - int(1 // dt) - left_idx), corr_row_row)
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlabel('timeline, ms')
    ax.set_title('Correlation of inner and outer sums (0-7 vs. 8-15)')
    plt.grid(True)
    plt.show()
    fig.savefig("./in_out_sums_corr_8.png", format="png")


#####################################
# Вывод проекций
#####################################
def projection_plot_ts(data_values, ts_1, ts_2, t_start, dt):
    """
    Вывод двух проекций и разницы между ними
    :param data_values:
    :param ts_1: Временная отметка из интервала 1
    :param ts_2: Временная отметка из интервала 2
    :param t_start:
    :param dt:
    :return:
    """
    idx_1 = int((ts_1 - t_start) // dt)
    idx_2 = int((ts_2 - t_start) // dt)
    fig, ax = plt.subplots(1, 3, figsize=(9, 4),
                        subplot_kw={'xticks': [2, 4, 6, 8, 10, 12, 14, 16],
                                    'yticks': [2, 4, 6, 8, 10, 12, 14, 16]})
    ax[0].imshow(data_values[:, :, idx_1], origin='lower')
    ax[0].set_title(f'{ts_1} ms')
    ax[1].imshow(data_values[:, :, idx_2], origin='lower')
    ax[1].set_title(f'{ts_2} ms')
    im = ax[2].imshow(np.fabs(np.subtract(data_values[:, :, idx_2], data_values[:, :, idx_1])), origin='lower')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    ax[2].set_title(f'Subtraction')
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.show()
    fig.savefig("./proj_diff.png", format="png")


#####################################
# Поиск центра масс
#####################################
def cm_slice(slice):
    sum_val = np.sum(slice, axis=(0, 1))
    x_c = np.arange(16).dot(np.sum(slice, axis=0)) / sum_val
    y_c = np.arange(16)[::-1].dot(np.sum(slice, axis=1)) / sum_val
    return x_c, y_c


def plot_cm(data_values, left_idx, right_idx):
    x = []
    y = []
    for i in range(left_idx, right_idx):
        x_c, y_c = cm_slice(data_values[:, :, i])
        x.append(x_c)
        y.append(y_c)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, ax = plt.subplots()
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, len(x))
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    # Set the values used for colormapping
    lc.set_array(np.arange(len(x)))
    lc.set_linewidth(2)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    ax.plot(x, y)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    plt.show()


#####################################
# Поиск центра пятна
#####################################
def center_slice(slice, k):
    avg = np.sum(slice, axis=(0, 1)) / (16 * 16)
    args = np.argwhere(slice > avg * k)
    x_c = np.mean(args[:, 1])
    y_c = np.mean(np.subtract(15, args[:, 0]))
    return x_c, y_c


def plot_center(data_values, left_idx, right_idx, k):
    x_l = []
    y_l = []
    for i in range(left_idx, right_idx):
        x_c, y_c = center_slice(data_values[:, :, i], k)
        x_l.append(x_c)
        y_l.append(y_c)

    x = np.array(x_l)
    y = np.array(y_l)
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    fig, ax = plt.subplots()
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(0, len(x))
    lc = LineCollection(segments, cmap='cool', norm=plt.Normalize(0, 16))
    # Set the values used for colormapping
    lc.set_array(np.linspace(0, 20, len(x)))
    lc.set_linewidth(3)
    line = ax.add_collection(lc)
    fig.colorbar(line, ax=ax)
    ax.plot(x, y, c='white', alpha=0.001)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_title(f'Brightness movement, coeff = {k}')
    plt.grid(True)
    # ax.autoscale()
    plt.show()


def optical_flow(data_values, idx_2, idx_1):
    im1 = np.zeros((16, 16, 1))
    im2 = np.zeros((16, 16, 1))
    im1[:, :, 0] = data_values[:, :, idx_1]
    im2[:, :, 0] = data_values[:, :, idx_2]
    # im1[:, :, 1] = 0
    # im2[:, :, 1] = 0
    # im1[:, :, 2] = 1
    # im2[:, :, 2] = 1
    # im1 = im1.astype(float) / 255.
    # im2 = im2.astype(float) / 255.

    alpha = 0.012
    ratio = 0.75
    minWidth = 4
    nOuterFPIterations = 7
    nInnerFPIterations = 1
    nSORIterations = 30
    colType = 1
    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        im1, im2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, im1.shape[0], im1.shape[1], im1.shape[2]))
    flow = np.concatenate((u[..., None], v[..., None]), axis=2)
    np.save('./outFlow.npy', flow)
    plt.imshow(flow[:, :, 0], interpolation='nearest')
    plt.show()
    plt.imshow(flow[:, :, 1], interpolation='nearest')
    plt.show()

    hsv = np.zeros(im1.shape, dtype=np.uint8)
    hsv[:, :, 0] = 255
    # hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # cv2.imwrite('./outFlow_new.png', hsv)
    plt.imshow(hsv[:, :, 0], interpolation='nearest')
    plt.show()
    #cv2.imwrite('./car2Warped_new.jpg', im2W[:, :, ::-1] * 255)


def main():
    # Загрузка данных из файла
    dt, t_start, t_end, data_values, rows_num, col_num = load_data("./37000_SPD16x16.mat")

    # Начало и конец рассматриваемого интервала
    interval_left_sec = 135
    interval_right_sec = 175

    # Нахождение индексов, соответствующих началу и концу интервала
    interval_left_idx = int((160.048 - t_start) // dt)
    interval_right_idx = int((167.3 - t_start) // dt)

    # draw_sum_gr(data_values, interval_left_idx, interval_right_idx)
    # draw_sum_gr_normed(data_values, interval_left_idx, interval_right_idx)
    # row_correlation(data_values, interval_left_idx, interval_right_idx, dt)
    # projection_plot_ts(data_values, 160.048, 167.3, t_start, dt)
    # in_out_sums_corr(data_values, interval_left_idx, interval_right_idx, dt)
    # plot_center(data_values, interval_left_idx, interval_right_idx, 1.3)
    optical_flow(data_values, interval_left_idx, interval_right_idx)


main()
