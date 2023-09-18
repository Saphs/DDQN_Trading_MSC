import numpy as np
import matplotlib.pyplot as plt

def l1_loss(x):
    return np.abs(x)

def smooth_l1_loss(x):
    return np.where(np.abs(x) < 1, 0.5 * x**2, np.abs(x) - 0.5)

def l2_loss(x):
    return x**2

x = np.linspace(-3, 3, 400)

l1_loss_values = l1_loss(x)
smooth_l1_loss_values = smooth_l1_loss(x)
l2_loss_values = l2_loss(x)

plt.figure(figsize=(4, 4), dpi=300)
plt.plot(x, l1_loss_values, label='L1 (MAE) Loss')
plt.plot(x, smooth_l1_loss_values, label='Smooth L1 Loss')
plt.plot(x, l2_loss_values, label='L2 (MSE) Loss')


plt.xlabel('x = |y_target - y|')
plt.ylabel('Loss')
plt.title('Vergleich der Kostenfunktionen')
plt.legend()
plt.tight_layout()
plt.grid()

plt.show()