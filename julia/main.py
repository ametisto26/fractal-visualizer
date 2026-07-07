import numpy as np
from numba import njit
import matplotlib.pyplot as plt

# ズーム履歴
history = []

# お気に入り保存
favorites = []

@njit(cache=True)
def compute_julia(
    cr,
    ci,
    cx,
    cy,
    scale,
    width,
    height,
):
    max_iter = int(300 + 50 * np.log2(scale + 1))

    image = np.zeros((height, width), dtype=np.float64)

    base_width = 3.0
    base_height = 3.0

    x_width = base_width / scale
    y_height = base_height / scale

    xmin = cx - x_width / 2
    xmax = cx + x_width / 2
    ymin = cy - y_height / 2
    ymax = cy + y_height / 2

    dx = (xmax - xmin) / (width - 1)
    dy = (ymax - ymin) / (height - 1)

    for py in range(height):

        imag = ymin + py * dy

        for px in range(width):

            real = xmin + px * dx

            zr = real
            zi = imag

            value = 0.0

            for i in range(max_iter):

                zr2 = zr * zr
                zi2 = zi * zi

                if zr2 + zi2 > 4.0:

                    r = np.sqrt(zr2 + zi2)

                    value = (
                        i
                        + 1
                        - np.log(np.log(r)) / np.log(2.0)
                    )

                    break

                zi = 2.0 * zr * zi + ci
                zr = zr2 - zi2 + cr

            image[py, px] = value

    return image

def julia(c, cx=0.0, cy=0.0, scale=1.0, width=1200, height=1200):

    # 描画範囲
    base_width = 3.0
    base_height = 3.0

    x_width = base_width / scale
    y_height = base_height / scale

    xmin = cx - x_width / 2
    xmax = cx + x_width / 2
    ymin = cy - y_height / 2
    ymax = cy + y_height / 2

    # 発散回数記録用
    divergence_step = compute_julia(
        c.real,
        c.imag,
        cx,
        cy,
        scale,
        width,
        height,
    )

    # 描画
    fig, ax = plt.subplots(figsize=(10, 10))

    im = ax.imshow(
        divergence_step,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        cmap="twilight_shifted"
    )

    plt.colorbar(im, ax=ax)

    ax.set_title(f"Julia Explorer | c={c}")

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return

        new_c = complex(event.xdata, event.ydata)
        plt.close()

        history.append((c, cx, cy, scale))

        julia(new_c, cx, cy, scale)

    def onkey(event):
        if event.key == "backspace" and history:
            prev_c, prev_cx, prev_cy, prev_scale = history.pop()
            plt.close()
            julia(prev_c, prev_cx, prev_cy, prev_scale)
        
        elif event.key == "s":

            favorites.append(c)
            print(f"Saved favorite: {c}")



    def on_scroll(event):

        if event.button == "up":
            new_scale = scale * 1.2
        elif event.button == "down":
            new_scale = scale / 1.2
        else:
            return

        plt.close()
        julia(c, cx, cy, new_scale, width, height)


    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    fig.canvas.mpl_connect("scroll_event", on_scroll)

    plt.show()

# c = 0.4 + 0.24j
# c = - 0.75 + 0.1j
c = - 0.743643887 + 0.131825904j

julia(c)
for i, z in enumerate(favorites):
    print(i, z)
