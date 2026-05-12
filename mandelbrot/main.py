import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 1, 800)
y = np.linspace(-1.5, 1.5, 800)

X, Y = np.meshgrid(x, y)
C = X + 1j * Y

Z = np.zeros_like(C)

for _ in range(50):
    Z = Z**2 + C

plt.imshow(np.abs(Z) < 2, extent=(-2,1,-1.5,1.5))
plt.show()