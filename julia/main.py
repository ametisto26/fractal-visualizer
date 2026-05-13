import numpy as np
import matplotlib.pyplot as plt

# ズーム履歴
history = []

def julia(c, cx=0.0, cy=0.0, scale=1.0, width=1200, height=1200):

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

    # 複素平面
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)

    X, Y = np.meshgrid(x, y)
    Z = X + 1j * Y

    # 発散回数記録用
    divergence_step = np.full(Z.shape, max_iter, dtype=float)

    #初期化
    # divergence_step[:] = max_iter

    # まだ発散していない点
    mask = np.ones(Z.shape, dtype=bool)

    for i in range(max_iter):

        # 未発散点だけ更新
        Z[mask] = Z[mask]**2 + c

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
    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(
        divergence_step,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap="twilight_shifted"
    )

    plt.colorbar(im, ax=ax)

    title = ax.set_title(f"Julia Explorer | c={c}")

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return

        new_c = complex(event.xdata, event.ydata)
        plt.close()

        history.append((c, cx, cy, scale))

        julia(new_c, cx, cy, scale)

    def onkey(event):
        if event.key == "backspace" and history:
            c, cx, cy, scale = history.pop()
            plt.close()
            julia(c, cx, cy, scale)


    def on_scroll(event):

        if event.button == "up":
            new_scale = scale * 1.2
        elif event.button == "down":
            new_scale = scale / 1.2

        plt.close()
        julia(c, cx, cy, new_scale, width, height)


    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    plt.show()

c = 0.4 + 0.24j

julia(c)
