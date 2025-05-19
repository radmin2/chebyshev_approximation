import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

def chebyshev_approximation(f, n):
    """
    Возвращает функцию p(x) -- приближение f(x) полиномом Чебышёва степени n.
    f: callable функция, определённая на [-1,1].
    n: степень полинома (n+1 коэффициентов).
    """
    # Выбор узлов Чебышёва-Лобато для построения сетки
    N = 2*(n+1)
    x = np.cos(np.linspace(0, np.pi, N))
    y = f(x)
    # Нахождение коэффициентов Чебышёва
    cheb = Chebyshev.fit(x, y, deg=n)
    return cheb  # объект, реализующий функцию p(x)

# Пример: аппроксимация sin(x) полиномом степени 5
f = np.sin
p5 = chebyshev_approximation(f, 5)
xs = np.linspace(-1, 1, 1000)
error = np.max(np.abs(f(xs) - p5(xs)))
print("Макс. погрешность sin(x), n=5:", error)

# Стабильность: добавим шум ±eps к коэффициентам и посмотрим ошибку
n = 10
p10 = chebyshev_approximation(np.sin, n)
coeffs = p10.coef.copy()
noise_levels = np.logspace(-9, -3, 7)
errors = []
for eps in noise_levels:
    noisy = coeffs + (np.random.rand(len(coeffs))*2-1)*eps
    # Оценить функцию с шумными коэффициентами
    p_noisy = Chebyshev(noisy, domain=[-1,1])
    err = np.max(np.abs(np.sin(xs) - p_noisy(xs)))
    errors.append(err)
print("Ошибка vs шум в коэффициентах:", list(zip(noise_levels, errors)))


import matplotlib.pyplot as plt

funcs = {
    'sin(x)': np.sin,
    'cos(x)': np.cos,
    'noisy sin': lambda x: np.sin(x) + np.random.uniform(-1e-6,1e-6, size=x.shape),
    'poly': lambda x: 0.5*x**5 - x**3 + 2*x**2 - x + 1
}
ns = list(range(2, 51)) + [256]
for name, f in funcs.items():
    E = []
    for n in ns:
        p = chebyshev_approximation(f, n)
        err = np.max(np.abs(f(xs) - p(xs)))
        E.append(err)
    plt.figure(figsize=(4,3))
    plt.semilogy(ns, E, marker='o', linestyle='-')
    plt.title(f'Ошибка аппроксимации {name}')
    plt.xlabel('n (степень полинома)')
    plt.ylabel('макс. ошибка')
    plt.grid(True)
    plt.savefig("graph_approximation_with_noise", dpi=300)
    plt.show()



# Построение графиков аппроксимации
import matplotlib.pyplot as plt

for name, f in funcs.items():
    for n in [5, 10, 20]:
        p = chebyshev_approximation(f, n)
        plt.figure(figsize=(4,3))
        plt.plot(xs, f(xs), 'b', label='f(x)')
        plt.plot(xs, p(xs), 'r--', label=f'p_{n}(x)')
        plt.title(f'{name}, n={n}')
        plt.legend()
        plt.ylim(-3,3)
        plt.grid(True)
        plt.savefig(f"{name}_approximation_with_{n}_coefficients.png", dpi=300)
        plt.show()
