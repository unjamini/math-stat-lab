import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math
import pandas
from prettytable import PrettyTable


len_list = [20, 100]
distr_type = ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']

def get_quartil(distr, p):
    n = len(distr)
    sorted = np.sort(distr)
    return sorted[int(np.floor(n * p) + np.ceil((n * p) - int(n * p)))]


def get_distr_samples(d_name, num):
    if d_name == 'Normal':
        return np.random.normal(0, 1, num)
    elif d_name == 'Cauchy':
        return np.random.standard_cauchy(num)
    elif d_name == 'Laplace':
        return np.random.laplace(0, math.sqrt(2) / 2, num)
    elif d_name == 'Poisson':
        return np.random.poisson(10, num)
    elif d_name == 'Uniform':
        return np.random.uniform(-math.sqrt(3), math.sqrt(3), num)
    return []


for dist_name in distr_type:
    data1 = get_distr_samples(dist_name, 20)
    q_20_1 = get_quartil(data1, 0.25)
    q_20_3 = get_quartil(data1, 0.75)
    x_20_1 = q_20_1 - 1.5 * (q_20_3 - q_20_1)
    x_20_2 = q_20_3 + 1.5 * (q_20_3 - q_20_1)
    print(f"Концы усов для 20 семплов: {x_20_1} и {x_20_2}")
    data1_more = data1[data1 > x_20_2]
    data1_less = data1[data1 < x_20_1]
    print(data1_more)
    print(data1_less)

    data2 = get_distr_samples(dist_name, 100)
    q_100_1 = get_quartil(data2, 0.25)
    q_100_3 = get_quartil(data2, 0.75)
    x_100_1 = q_100_1 - 1.5 * (q_100_3 - q_100_1)
    x_100_2 = q_100_3 + 1.5 * (q_100_3 - q_100_1)
    print(f"Концы усов для 100 семплов: {x_100_1} и {x_100_2}")
    data2_more = data2[data2 > x_100_2]
    data2_less = data2[data2 < x_100_1]
    print(data2_more)
    print(data2_less)


    print(f"{dist_name} для 20:\n"
          f"{(len(data1_less) + len(data1_more))/ 20}\n"
          f"{dist_name} для 100:\n"
          f"{(len(data2_less) + len(data2_more))/ 100}\n")
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title(dist_name + ' distribution. 20 samples')
    ax1.boxplot(data1)
    ax2.set_title(dist_name + ' distribution. 100 samples')
    ax2.boxplot(data2)
    # plt.show()





