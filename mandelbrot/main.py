import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(cx, cy, scale=1.0, width=800, height=800, max_iter=100):

    # 描画範囲
    base_width = 3.0
    base_height = 3.0

    x_width = base_width / scale
    y_height = base_height / scale

    xmin = cx - x_width / 2
    xmax = cx + x_width / 2
    ymin = cy - y_height / 2
    ymax = cy + y_height / 2

    # 解像度
    # width, height：引数をそのまま使う

    # 最大反復回数
    # max_iter：引数をそのまま使う

    # 複素平面
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    # 初期値
    Z = np.zeros_like(C)

    # 発散回数記録用
    divergence_step = np.zeros(C.shape, dtype=float)

    #初期化
    divergence_step[:] = max_iter

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
        abs_z = np.abs(Z[newly_diverged]) + 1e-12

        nu = i + 1 - np.log(np.log(abs_z)) / np.log(2.0)

        divergence_step[newly_diverged] = nu

        # 発散済みを除外
        mask &= ~diverged

    # 描画
    plt.figure(figsize=(8, 8))

    plt.imshow(
        divergence_step,
        extent = [xmin, xmax, ymin, ymax],
        cmap = "twilight_shifted",
        origin = "lower",
        interpolation = "bicubic"
    )

    plt.colorbar(label="Iterations")

    plt.title("Mandelbrot Set")
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")

    plt.show()

mandelbrot(-0.75, 0.1, scale=10)
