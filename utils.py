import matplotlib.pyplot as plt
import matplotlib.patches as patches


def plot_bbox(img, bboxes):
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    for rect in bboxes:
        ax.add_patch(patches.Rectangle((rect[0], rect[1]), rect[2] - rect[0] + 1, rect[3] - rect[1] + 1, linewidth=2,
                                   edgecolor='green', fill=False))
        plt.text(rect[0], rect[1], '%d' % int(rect[4]), c='red', fontsize=12)
    plt.imshow(img)
    plt.pause(0.01)
    ax.clear()