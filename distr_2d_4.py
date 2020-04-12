import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math
from scipy.special import factorial
from prettytable import PrettyTable


rho_list = [0, 0.5, 0.9]
len_list = [20, 60, 100]

mixed_10 = 0.9 * np.random.multivariate_normal([0, 0],[[1, 0],[0, 1]], size=10) \
                  + 0.1 * np.random.multivariate_normal([0, 0],[[10, 0],[0, 10]], size=10)


def fechner_c(x, y):
    x_0 = x - np.mean(x)
    y_0 = y - np.mean(y)
    r_q = np.sum(np.multiply(np.sign(x_0), np.sign(y_0))) / x_0.shape
    return r_q



for d_num in len_list:
    resTable = PrettyTable()
    resTable.float_format = "2.2"
    resTable.field_names = ["Characteristic", "Pearson", "Spearman", "Quad"]
    for rho in rho_list:
        pearson = []
        spearman = []
        quad = []
        for it in range(1000):
            sample_d = np.random.multivariate_normal([0, 0],[[1, rho],[rho, 1]], size=d_num)
            pearson.append(stats.pearsonr(sample_d[:, 0], sample_d[:, 1]))
            spearman.append(stats.spearmanr(sample_d[:, 0], sample_d[:, 1]))
            quad.append(fechner_c(sample_d[:, 0], sample_d[:, 1]))
        resTable.add_row([str(d_num) + " E(z) " + str(rho),
                        np.around(np.mean(pearson), decimals=6),
                        np.around(np.mean(spearman), decimals=6),
                        np.around(np.mean(quad), decimals=6)])
        resTable.add_row([str(d_num) + " E(z^2) " + str(rho),
                          np.around(np.mean(np.square(pearson)), decimals=6),
                          np.around(np.mean(np.square(spearman)), decimals=6),
                          np.around(np.mean(np.square(quad)), decimals=6)])

        resTable.add_row([str(d_num) + " D(z) " + str(rho),
                          np.around(np.std(pearson) * np.std(pearson), decimals=6),
                          np.around(np.std(spearman) * np.std(spearman), decimals=6),
                          np.around(np.std(quad) * np.std(quad), decimals=6)])
    print(resTable)

resTable = PrettyTable()
resTable.float_format = "2.2"
resTable.field_names = ["Characteristic", "Pearson", "Spearman", "Quad"]
for d_num in len_list:
    pearson = []
    spearman = []
    quad = []
    for it in range(1000):
        sample_d = 0.9 * np.random.multivariate_normal([0, 0],[[1, 0.9],[0.9, 1]], size=d_num) \
                  + 0.1 * np.random.multivariate_normal([0, 0],[[10, -0.9],[-0.9, 10]], size=d_num)
        sample_d_sorted = np.sort(sample_d)
        pearson.append(stats.pearsonr(sample_d[:, 0], sample_d[:, 1]))
        spearman.append(stats.spearmanr(sample_d[:, 0], sample_d[:, 1]))
        quad.append(fechner_c(sample_d[:, 0], sample_d[:, 1]))
    resTable.add_row([str(d_num) + " E(z) mixed" + str(rho),
                        np.around(np.mean(pearson), decimals=6),
                        np.around(np.mean(spearman), decimals=6),
                        np.around(np.mean(quad), decimals=6)])
    resTable.add_row([str(d_num) + " E(z^2) mixed" + str(rho),
                          np.around(np.mean(np.square(pearson)), decimals=6),
                          np.around(np.mean(np.square(spearman)), decimals=6),
                          np.around(np.mean(np.square(quad)), decimals=6)])

    resTable.add_row([str(d_num) + " D(z) mixed" ,
                          np.around(np.std(pearson) * np.std(pearson), decimals=6),
                          np.around(np.std(spearman) * np.std(spearman), decimals=6),
                          np.around(np.std(quad) * np.std(quad), decimals=6)])
    print(resTable)