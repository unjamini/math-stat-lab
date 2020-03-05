import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math
import statsmodels.api as sm
from scipy.special import factorial


len_list = [20, 60, 100]
distr_type = ['Normal', 'Cauchy', 'Laplace', 'Poisson', 'Uniform']


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


def get_cdf_vals(x, d_name):
    if d_name == 'Normal':
        return stats.norm.cdf(x, 0, 1)
    elif d_name == 'Cauchy':
        return stats.cauchy.cdf(x, 0, 1)
    elif d_name == 'Laplace':
        return stats.laplace.cdf(x, 0, math.sqrt(2) / 2)
    elif d_name == 'Poisson':
        return stats.poisson.cdf(x, 10)
    elif d_name == 'Uniform':
        return stats.uniform.cdf(x, -math.sqrt(3), 2 * math.sqrt(3))
    return []


def get_pdf_vals(x, d_name):
    if d_name == 'Normal':
        return stats.norm.pdf(x, 0, 1)
    elif d_name == 'Cauchy':
        return stats.cauchy.pdf(x, 0, 1)
    elif d_name == 'Laplace':
        return stats.laplace.pdf(x, 0, math.sqrt(2) / 2)
    elif d_name == 'Poisson':
        return np.exp(-10) * np.power(10, x)/factorial(x)
    elif d_name == 'Uniform':
        return stats.uniform.pdf(x, -math.sqrt(3), 2 * math.sqrt(3))
    return []


def cde_plots():
    samples = [[], [], []]
    for dist_name in distr_type:
        fig, ax = plt.subplots(1, 3)
        for i in range(len(len_list)):
            if dist_name == 'Poisson':
                r = (6, 14)
            else:
                r = (-4, 4)
            x = np.linspace(r[0], r[1], 1000)
            samples[i] = get_distr_samples(dist_name, len_list[i])
            ecdf = sm.distributions.ECDF(samples[i])
            y = ecdf(x)
            ax[i].plot(x, y, color='m', label='Empirical distribution function')
            y = get_cdf_vals(x, dist_name)
            ax[i].plot(x, y, color='orange', label='Distribution function')
            ax[i].set_title(dist_name + '\n n = ' + str(len_list[i]))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 6})
        plt.show()
        fig.savefig(dist_name + '_cde.png', dpi=200)


for dist_name in distr_type:
    for i in range(len(len_list)):
        fig, ax = plt.subplots(1, 3)
        if dist_name == 'Poisson':
            r = (6, 14)
        else:
            r = (-4, 4)
        x = np.linspace(r[0], r[1], 1000)
        y = get_pdf_vals(x, dist_name)
        samples = get_distr_samples(dist_name, len_list[i])
        samples = samples[samples <= r[1]]
        samples = samples[samples >= r[0]]
        kde = stats.gaussian_kde(samples)
        kde.set_bandwidth(bw_method='silverman')
        h_n = kde.factor
        sns.kdeplot(samples, ax=ax[0], bw=h_n/2, color='m')
        ax[0].set_title(r'$h = \frac{h_n}{2}$')
        ax[0].plot(x, y, color='orange')
        ax[1].plot(x, y, color='orange')
        ax[2].plot(x, y, color='orange', label='Real density function')
        sns.kdeplot(samples, ax=ax[1], bw=h_n, color='m')
        ax[1].set_title(r'$h = h_n$')
        sns.kdeplot(samples, ax=ax[2], bw=2*h_n, color='m', label='Kernel density esimation')
        ax[2].set_title(r'$h = 2 * h_n$')
        fig.suptitle(dist_name + ' n = ' + str(len_list[i]))
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 6})
        plt.show()
        fig.savefig(dist_name + str(len_list[i]) + '_kde.png', dpi=200)