import numpy as np
import matplotlib.pyplot as plt

# ズーム履歴
history = []

def burning_ship(cx, cy, scale=1.0, width=1200, height=1200):

    # 反復回数
    max_iter = int(100 + 20 * np.log2(scale + 1))

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
        Z_real = Z[mask].real
        Z_imag = np.abs(Z[mask].imag)

        Z[mask] = (Z_real + 1j * Z_imag)**2 + C[mask]

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

    # 最後まで発散しなかった点
    divergence_step[mask] = 0

    # 描画
    plt.figure(figsize=(8, 8))

    extent = [xmin, xmax, ymin, ymax]

    img = plt.imshow(
        divergence_step,
        extent = extent,
        cmap = "viridis",
        origin = "lower",
        interpolation = "bicubic"
    )

    # クリックでズーム機能
    def onclick(event):

        if event.xdata is None or event.ydata is None:
            return

        history.append((cx, cy, scale))

        new_cx = event.xdata
        new_cy = event.ydata

        plt.close()

        burning_ship(
            new_cx,
            new_cy,
            scale=scale * 2,
            width=width,
            height=height
        )

    # キーボードで戻る機能
    def on_key(event):

        if event.key == "backspace" and len(history) > 0:

            cx, cy, scale = history.pop()

            plt.close()

            burning_ship(
                cx,
                cy,
                scale=scale,
                width=width,
                height=height
            )

    fig = plt.gcf()
    
    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", on_key)

    plt.colorbar(label="Iterations")

    plt.title(
        f"Burning Ship Fractal "
        f"(scale={scale:.2f}, max_iter={max_iter})"
    )
    plt.xlabel("Re(c)")
    plt.ylabel("Im(c)")

    plt.show()

burning_ship(-0.5, -0.5, scale=1)
