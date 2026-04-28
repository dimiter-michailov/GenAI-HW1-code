import numpy as np
import matplotlib.pyplot as plt

n = 5000
true_expected_value = 2.0

init_samples = np.random.uniform(0, 1, n)
samples = 3* np.sqrt(init_samples)

cumsum = np.cumsum(samples)
iterations = np.arange(1, n + 1)
running_average = cumsum / iterations

plt.figure(figsize=(10, 6))
plt.plot(iterations, running_average, label='Monte Carlo Approximation', color='blue')
plt.axhline(y=true_expected_value, color='red', linestyle='--', label=f'True Expected Value = {true_expected_value}')

plt.title('Monte Carlo Convergence of E[X]')
plt.xlabel('Number of Simulations (N)')
plt.ylabel('Estimated Expected Value')
plt.xlim(0, n)
plt.ylim(1.5, 2.5) # Zoomed in a bit to see convergence clearly
plt.grid(True, alpha=0.3)
plt.legend()

plt.show()

# Print the final approximation
print(f"Analytical Expected Value: {true_expected_value}")
print(f"Monte Carlo Approximation (N={n}): {running_average[-1]:.4f}")