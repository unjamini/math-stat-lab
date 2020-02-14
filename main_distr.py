import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math

x = np.linspace(-10, 10, 1000)
# distr = 'Cauchy'
# distr = 'Norm'
# distr = 'Laplace'
# distr = 'Poisson'
distr = 'Uniform'


if distr == 'Norm':
    x = np.linspace(-3, 3, 1000)
    normal_distr_10 = np.random.normal(0, 1, 10)
    normal_distr_50 = np.random.normal(0, 1, 50)
    normal_distr_1000 = np.random.normal(0, 1, 1000)

    f = plt.figure(1, figsize=(9, 3))
    plt.subplot(131)
    sns.distplot(normal_distr_10, color="g")
    plt.plot(x, stats.norm.pdf(x, 0, 1), color="orange")
    plt.title('10 значений')

    plt.subplot(132)
    sns.distplot(normal_distr_50, color="g")
    plt.plot(x, stats.norm.pdf(x, 0, 1), color="orange")
    plt.title('50 значений')


    plt.subplot(133)
    sns.distplot(normal_distr_1000, color="g", label='Экспериментальные данные')
    plt.plot(x, stats.norm.pdf(x, 0, 1), color="orange", label='Реальные данные')
    plt.title('1000 значений')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 6})
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=2.0)
    plt.show()
    f.savefig('Normal_distr.png', dpi=200)

if distr == 'Cauchy':
    x = np.linspace(-10, 10, 1000)
    cauchy_distr_10 = np.random.standard_cauchy(10)
    cauchy_distr_50 = np.random.standard_cauchy(50)
    cauchy_distr_1000 = np.random.standard_cauchy(1000)

    f = plt.figure(1, figsize=(9, 3))
    plt.subplot(131)
    sns.distplot(cauchy_distr_10, color="g")
    plt.plot(x, stats.cauchy.pdf(x, 0, 1), color="orange")
    plt.title('10 значений')

    plt.subplot(132)
    sns.distplot(cauchy_distr_50, color="g")
    plt.plot(x, stats.cauchy.pdf(x, 0, 1), color="orange")
    plt.title('50 значений')


    plt.subplot(133)
    sns.distplot(cauchy_distr_1000, color="g", label='Экспериментальные данные')
    plt.plot(x, stats.cauchy.pdf(x, 0, 1), color="orange", label='Реальные данные')
    plt.title('1000 значений')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 6})
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=2.0)
    plt.show()
    #f.savefig('Cauchy_distr.png', dpi=200)

if distr == 'Laplace':
    x = np.linspace(-4, 4, 1000)
    lapl_distr_10 = np.random.laplace(0, math.sqrt(2) / 2, 10)
    lapl_distr_50 = np.random.laplace(0, math.sqrt(2) / 2, 50)
    lapl_distr_1000 = np.random.laplace(0, math.sqrt(2) / 2, 1000)

    f = plt.figure(1, figsize=(9, 3))
    plt.subplot(131)
    sns.distplot(lapl_distr_10, color="g")
    plt.plot(x, stats.laplace.pdf(x, math.sqrt(2) / 2, 1), color="orange")
    plt.title('10 значений')

    plt.subplot(132)
    sns.distplot(lapl_distr_50, color="g")
    plt.plot(x, stats.laplace.pdf(x, math.sqrt(2) / 2, 1), color="orange")
    plt.title('50 значений')

    plt.subplot(133)
    sns.distplot(lapl_distr_1000, color="g", label='Экспериментальные данные')
    plt.plot(x, stats.laplace.pdf(x, math.sqrt(2) / 2, 1), color="orange", label='Реальные данные')
    plt.title('1000 значений')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 6})
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=2.0)
    plt.show()
    # f.savefig('Laplace_distr.png', dpi=200)

if distr == 'Poisson':
    x = np.linspace(-4, 4, 1000)
    poi_distr_10 = np.random.poisson(10, 10)
    poi_distr_50 = np.random.poisson(10, 50)
    poi_distr_1000 = np.random.poisson(10, 1000)

    f = plt.figure(1, figsize=(9, 3))
    plt.subplot(131)
    sns.distplot(poi_distr_10, color="g")
    plt.plot(x, stats.poisson.pmf(x, 10, 0), color="orange")
    plt.title('10 значений')

    plt.subplot(132)
    sns.distplot(poi_distr_50, color="g")
    plt.plot(x, stats.poisson.pmf(x, 10, 0), color="orange")
    plt.title('50 значений')

    plt.subplot(133)
    sns.distplot(poi_distr_1000, color="g", label='Экспериментальные данные')
    plt.plot(x, stats.poisson.pmf(x, 10, 0), color="orange", label='Реальные данные')
    plt.title('1000 значений')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 6})
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=2.0)
    plt.show()
    # f.savefig('Poisson_distr.png', dpi=200)

if distr == 'Uniform':
    x = np.linspace(-4, 4, 1000)
    uni_distr_10 = np.random.uniform(-math.sqrt(3), math.sqrt(3), 10)
    uni_distr_50 = np.random.uniform(-math.sqrt(3), math.sqrt(3), 50)
    uni_distr_1000 = np.random.uniform(-math.sqrt(3), math.sqrt(3), 1000)

    f = plt.figure(1, figsize=(9, 3))
    plt.subplot(131)
    sns.distplot(uni_distr_10, color="g")
    plt.plot(x, stats.uniform.pdf(x, -math.sqrt(3), 2 * math.sqrt(3)), color="orange")
    plt.title('10 значений')

    plt.subplot(132)
    sns.distplot(uni_distr_50, color="g")
    plt.plot(x, stats.uniform.pdf(x, -math.sqrt(3), 2 * math.sqrt(3)), color="orange")
    plt.title('50 значений')

    plt.subplot(133)
    sns.distplot(uni_distr_1000, color="g", label='Экспериментальные данные')
    plt.plot(x, stats.uniform.pdf(x, -math.sqrt(3), 2 * math.sqrt(3)), color="orange", label='Реальные данные')
    plt.title('1000 значений')

    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., prop={'size': 6})
    plt.tight_layout(pad=1, w_pad=0.5, h_pad=2.0)
    plt.show()
    # f.savefig('Uniform_distr.png', dpi=200)