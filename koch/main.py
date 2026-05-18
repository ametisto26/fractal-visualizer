import numpy as np
import matplotlib.pyplot as plt

# 履歴
history = []


def rotate(v, theta):

       rot = np.array([
              [np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]
       ])

       return rot @ v


def koch(p1, p2, depth):

       if depth == 0:
              return [p1, p2]

       v = p2 - p1

       s = p1 + v / 3
       t = p1 + 2 * v / 3

       peak = s + rotate(v / 3, -np.pi / 3)

       a = koch(p1, s, depth - 1)
       b = koch(s, peak, depth - 1)
       c = koch(peak, t, depth - 1)
       d = koch(t, p2, depth - 1)

       return (
              a[:-1]
              + b[:-1]
              + c[:-1]
              + d
       )


def koch_snowflake(
       depth,
       xmin=-0.2,
       xmax=1.2,
       ymin=-0.2,
       ymax=1.1
       ):

       # 正三角形
       p1 = np.array([0.0, 0.0])
       p2 = np.array([1.0, 0.0])

       p3 = np.array([
              0.5,
              np.sqrt(3) / 2
       ])

       edge1 = koch(p1, p2, depth)
       edge2 = koch(p2, p3, depth)
       edge3 = koch(p3, p1, depth)

       points = (
              edge1[:-1]
              + edge2[:-1]
              + edge3
       )

       points = np.array(points)

       fig, ax = plt.subplots(figsize=(8, 8))

       ax.plot(points[:, 0], points[:, 1])

       ax.set_xlim(xmin, xmax)
       ax.set_ylim(ymin, ymax)

       ax.set_aspect("equal")
       ax.axis("off")

       ax.set_title(f"Koch Snowflake (depth={depth})")

       # クリックズーム
       def onclick(event):

              if event.xdata is None or event.ydata is None:
                     return

              history.append(
              (depth, xmin, xmax, ymin, ymax)
              )

              cx = event.xdata
              cy = event.ydata

              zoom = 1.7

              width = (xmax - xmin) / zoom
              height = (ymax - ymin) / zoom

              new_xmin = cx - width / 2
              new_xmax = cx + width / 2

              new_ymin = cy - height / 2
              new_ymax = cy + height / 2

              plt.close()

              koch_snowflake(
              depth + 1,
              new_xmin,
              new_xmax,
              new_ymin,
              new_ymax
              )

       # 戻る
       def on_key(event):

              if event.key == "backspace" and history:

                     state = history.pop()

              plt.close()

              koch_snowflake(*state)

       # スクロールズーム
       def on_scroll(event):

              if event.xdata is None or event.ydata is None:
                     return

              history.append(
                     (depth, xmin, xmax, ymin, ymax)
              )

              cx = event.xdata
              cy = event.ydata

              if event.button == "up":
                     zoom = 1.2
              elif event.button == "down":
                     zoom = 1 / 1.2
              else:
                     return

              width = (xmax - xmin) / zoom
              height = (ymax - ymin) / zoom

              new_xmin = cx - width / 2
              new_xmax = cx + width / 2

              new_ymin = cy - height / 2
              new_ymax = cy + height / 2

              plt.close()

              koch_snowflake(
                     depth,
                     new_xmin,
                     new_xmax,
                     new_ymin,
                     new_ymax
              )



       fig.canvas.mpl_connect(
              "button_press_event",
              onclick
       )

       fig.canvas.mpl_connect(
              "scroll_event",
              on_scroll
       )

       fig.canvas.mpl_connect(
              "key_press_event",
              on_key
       )

       plt.show()


koch_snowflake(1)
