import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev

def chebyshev_approximation(f, n):
    N = 2 * (n + 1)
    x = np.cos(np.linspace(0, np.pi, N))
    y = f(x)
    return Chebyshev.fit(x, y, deg=n)

f = lambda x: np.sin(x * 3 * np.pi)
xs = np.linspace(-1, 1, 1000)
f_true = f(xs)

# === График 1 ===
n_fixed = 30
p = chebyshev_approximation(f, n_fixed)
base_coeffs = p.coef.copy()

noise_levels = np.logspace(-12, -3, 20)
errors_noise_level = []

for eps in noise_levels:
    noisy_coeffs = base_coeffs + np.random.uniform(-eps, eps, size=base_coeffs.shape)
    p_noisy = Chebyshev(noisy_coeffs, domain=[-1, 1])
    error = np.max(np.abs(f_true - p_noisy(xs)))
    errors_noise_level.append(error)

plt.figure(figsize=(8, 5))
plt.loglog(noise_levels, errors_noise_level, marker='o')
plt.xlabel("Амплитуда шума в коэффициентах")
plt.ylabel("Максимальная ошибка аппроксимации")
plt.title("График 1: Ошибка vs уровень шума (n=30)")
plt.grid(True, which='both')
plt.savefig("graph1_error_vs_noise_level.png", dpi=300)
plt.show()

# === График 2 ===
noise_amplitude = 1e-6
ns = list(range(2, 101, 2))
errors_fixed_noise = []

for n in ns:
    p = chebyshev_approximation(f, n)
    coeffs = p.coef.copy()
    noisy_coeffs = coeffs + np.random.uniform(-noise_amplitude, noise_amplitude, size=coeffs.shape)
    p_noisy = Chebyshev(noisy_coeffs, domain=[-1, 1])
    error = np.max(np.abs(f(xs) - p_noisy(xs)))
    errors_fixed_noise.append(error)

plt.figure(figsize=(8, 5))
plt.plot(ns, errors_fixed_noise, marker='o')
plt.yscale('log')
plt.xlabel("Степень полинома n")
plt.ylabel("Максимальная ошибка при шуме ±1e-6")
plt.title("График 2: Ошибка vs степень полинома при фиксированном шуме")
plt.grid(True, which='both')
plt.savefig("graph2_error_vs_degree.png", dpi=300)
plt.show()

# === График 3 ===
error_ratios = []

for n in ns:
    p = chebyshev_approximation(f, n)
    coeffs = p.coef.copy()
    error_exact = np.max(np.abs(f(xs) - p(xs)))
    noisy_coeffs = coeffs + np.random.uniform(-noise_amplitude, noise_amplitude, size=coeffs.shape)
    p_noisy = Chebyshev(noisy_coeffs, domain=[-1, 1])
    error_noisy = np.max(np.abs(f(xs) - p_noisy(xs)))
    error_ratio = error_noisy / error_exact if error_exact > 0 else np.nan
    error_ratios.append(error_ratio)

plt.figure(figsize=(8, 5))
plt.plot(ns, error_ratios, marker='o')
plt.yscale('log')
plt.xlabel("Степень полинома n")
plt.ylabel("Ошибка с шумом / без")
plt.title("График 3: Рост чувствительности аппроксимации к шуму в коэффициентах")
plt.grid(True, which='both')
plt.savefig("graph3_error_ratio_vs_degree.png", dpi=300)
plt.show()
