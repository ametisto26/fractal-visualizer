import numpy as np
import matplotlib.pyplot as plt

# 描画範囲
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.5, 1.5

# 解像度
width, height = 800, 800

# 最大反復回数
max_iter = 100

# 複素平面
x = np.linspace(xmin, xmax, width)
y = np.linspace(ymin, ymax, height)

X, Y = np.meshgrid(x, y)
C = X + 1j * Y

# 初期値
Z = np.zeros_like(C)

# 発散回数記録用
divergence_step = np.zeros(C.shape, dtype=int)

# まだ発散していない点
mask = np.ones(C.shape, dtype=bool)

for i in range(max_iter):

    # 未発散点だけ更新
    Z[mask] = Z[mask]**2 + C[mask]

    # 発散判定
    diverged = np.abs(Z) > 2

    # 今回新しく発散した点
    newly_diverged = diverged & mask

    # 発散回数を記録
    divergence_step[newly_diverged] = i

    # 発散済みを除外
    mask &= ~diverged

# 描画
plt.figure(figsize=(8, 8))

plt.imshow(
    divergence_step,
    extent=[xmin, xmax, ymin, ymax],
    cmap="twilight_shifted",
    origin="lower"
)

plt.colorbar(label="Iterations")

plt.title("Mandelbrot Set")
plt.xlabel("Re(c)")
plt.ylabel("Im(c)")

plt.show()
