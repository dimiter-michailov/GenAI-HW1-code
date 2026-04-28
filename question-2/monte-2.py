import numpy as np
import matplotlib.pyplot as plt
import math

lam = 4

state_space = np.array([0, 1, 2, 3, 4, 5, 6])

M = 1 / (
    (4**0 * np.exp(-4) / math.factorial(0)) +
    (4**1 * np.exp(-4) / math.factorial(1)) +
    (4**2 * np.exp(-4) / math.factorial(2)) +
    (4**3 * np.exp(-4) / math.factorial(3)) +
    (4**4 * np.exp(-4) / math.factorial(4)) +
    2
)

distribution = np.array([
    M * 4**0 * np.exp(-4) / math.factorial(0),
    M * 4**1 * np.exp(-4) / math.factorial(1),
    M * 4**2 * np.exp(-4) / math.factorial(2),
    M * 4**3 * np.exp(-4) / math.factorial(3),
    M * 4**4 * np.exp(-4) / math.factorial(4),
    M,
    M
])

exact_mean = np.sum(state_space * distribution)

sample_sizes = [100, 200, 300, 10000]
estimated_means = []

for n in sample_sizes:
    samples = np.random.choice(state_space, size=n, p=distribution)
    estimated_mean = np.mean(samples)
    estimated_means.append(estimated_mean)

plt.plot(sample_sizes, estimated_means, marker='o')
plt.axhline(exact_mean, linestyle='--')
plt.xlabel("Number of samples")
plt.ylabel("Estimated expected value")
plt.title("Monte Carlo approximation of E[X]")
plt.show()
