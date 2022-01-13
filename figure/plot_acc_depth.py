from util.plot import plot, get_styles


if __name__ == '__main__':
    data = {
        'CNN': [53.7, 63.7, 71.4, 80.4, 84.3],
        'MHSA': [96.5, 96.9, 97.0, 97.4, 97.5],
        'NTN': [96.8, 97.3, 97.8, 97.7, 98.0],
        'MTN': [80.0, 81.6, 82.4, 91.5, 93.8],
        'MTN-Li': [58.3, 64.2, 77.1, 87.2, 86.4],
        'CMTN': [97.8, 98.3, 98.1, 98.4, 98.7],
        'STN': [47.2, 68.1, 85.2, 95.3, 95.6],
        'ETN': [34.6, 47.3, 38.7, 38.7, 48.9],
        'ATN': [96.9, 96.3, 96.6, 96.7, 96.3]
    }

    ys, legends = [], []
    for key, value in data.items():
        legends.append(key)
        ys.append(value)
    xs = [[i for i in range(1, 6)]] * len(ys)
    styles = get_styles(len(xs))
    plot(xs=xs, ys=ys, xlabel='Depth', ylabel='Acc (%)', legends=legends, styles=styles, loc='lower right',
         linewidth=1, markersize=6, filename='acc-depth.png')
