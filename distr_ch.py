import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math

from prettytable import PrettyTable


len_list = [10, 100, 1000]
distr_type = ['Norm', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']
# distr_type = ['Norm']


def get_quartil(distr, p):
    n = len(distr)
    sorted = np.sort(distr)
    return sorted[int(np.floor(n * p) + np.ceil((n * p) - int(n * p)))]



def get_distr_samples(d_name, num):
    if d_name == 'Norm':
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
    print(f"Распределение {dist_name}")
    resTable = PrettyTable()
    resTable.float_format = "2.2"
    resTable.field_names = ["Characteristic", "Mean", "Median", "Zr", "Zq", "Ztr"]
    for d_num in len_list:
        mean = []
        med = []
        z_r = []
        z_q = []
        z_tr = []
        for it in range(1000):
            sample_d = get_distr_samples(dist_name, d_num)
            sample_d_sorted = np.sort(sample_d)
            mean.append(np.mean(sample_d))
            med.append(np.median(sample_d))
            z_r.append((sample_d_sorted[0] + sample_d_sorted[-1]) / 2)
            z_q.append((get_quartil(sample_d, 0.25) + get_quartil(sample_d, 0.75)) / 2)
            z_tr.append(stats.trim_mean(sample_d, 0.25))
        resTable.add_row([dist_name + " E(z) " + str(d_num),
                          np.around(np.mean(mean), decimals=6),
                          np.around(np.mean(med), decimals=6),
                          np.around(np.mean(z_r), decimals=6),
                          np.around(np.mean(z_q), decimals=6),
                         np.around(np.mean(z_tr), decimals=6)])
        resTable.add_row([dist_name + " D(z) " + str(d_num),
                          np.around(np.std(mean) * np.std(mean), decimals=6),
                          np.around(np.std(med) * np.std(med), decimals=6),
                          np.around(np.std(z_r) * np.std(z_r), decimals=6),
                          np.around(np.std(z_q) * np.std(z_q), decimals=6),
                          np.around(np.std(z_tr) * np.std(z_tr), decimals=6)])

    print(resTable)