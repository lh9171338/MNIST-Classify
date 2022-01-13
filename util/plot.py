import random
import matplotlib.pyplot as plt


def get_styles(num, seed=0, colors=None, lines=None, marks=None):
    colors = colors or ['b', 'g', 'r', 'c', 'm', 'y']
    lines = lines or ['-', '--', ':']
    marks = marks or ['o', 'v', '^', 's', 'p', '*', 'h', 'D']

    styles = []
    random.seed(seed)
    for index in range(num):
        if index % len(colors) == 0:
            random.shuffle(colors)
        if index % len(lines) == 0:
            random.shuffle(lines)
        if index % len(marks) == 0:
            random.shuffle(marks)

        color_index = index % len(colors)
        line_index = index % len(lines)
        mark_index = index % len(marks)
        style = colors[color_index] + lines[line_index] + marks[mark_index]
        styles.append(style)

    return styles


def plot(xs, ys, xlabel, ylabel, styles, legends=None, ylim=None, fontsize=12, linewidth=2, markersize=10, filename=None, loc='upper right', show=False):

    plt.figure()
    plt.grid(True)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)

    for x, y, style in zip(xs, ys, styles):
        plt.plot(x, y, style, linewidth=linewidth, markersize=markersize)
    if legends is not None:
        plt.legend(legends, loc=loc)
    if filename is not None:
        plt.savefig(filename, format='png', bbox_inches='tight')
    if filename is None or show:
        plt.show()
