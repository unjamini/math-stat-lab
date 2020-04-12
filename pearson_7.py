import numpy as np
from scipy.stats import chi2
import scipy.stats as stats

samples = np.random.normal(0, 1, size=100)
mu_c = np.mean(samples)
sigma_c = np.std(samples)


def get_cdf(x, mu=mu_c, sigma=sigma_c):
    return stats.norm.cdf(x, mu, sigma)


alpha = 0.05
p = 1 - alpha
k = 7
left = mu_c - 3
right = mu_c + 3
borders = np.linspace(left, right, num=(k - 1))
value = chi2.ppf(p, k - 1)
p_arr = np.array([get_cdf(borders[0])])
for i in range(len(borders) - 1):
    val = get_cdf(borders[i + 1]) - get_cdf(borders[i])
    p_arr = np.append(p_arr, val)
p_arr = np.append(p_arr, 1 - get_cdf(borders[-1]))
print(f"Промежутки: {borders} \n"
      f"p_i: \n {p_arr} \n"
      f"n * p_i: \n {p_arr * 100} \n"
      f"Сумма: {np.sum(p_arr)}")
n_arr = np.array([len(samples[samples <= borders[0]])])
for i in range(len(borders) - 1):
    n_arr = np.append(n_arr, len(samples[(samples <= borders[i + 1]) & (samples >= borders[i])]))
n_arr = np.append(n_arr, len(samples[samples >= borders[-1]]))
res_arr = np.divide(np.multiply((n_arr - p_arr * 100), (n_arr - p_arr * 100)), p_arr * 100)
print(f"n_i: \n {n_arr} \n"
      f"Сумма: {np.sum(n_arr)}\n"
      f"n_i  - n*p_i: {n_arr - p_arr * 100}\n"
      f"res: {res_arr}\n"
      f"res_sum = {np.sum(res_arr)}\n")
