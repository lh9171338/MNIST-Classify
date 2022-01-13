from util.plot import plot, get_styles


if __name__ == '__main__':
    data = {
        'CNN': [0.6, 25.3, 9.4, 5.1, 76.2],
        'MHSA': [1.4, 84.3, 3.6, 13.3, 98.5],
        'NTN': [2.5, 318.9, 11.2, 23.6, 98.7],
        'MTN': [1.7, 318.9, 11.1, 66.5, 95.8],
        'MTN-Li': [1.7, 66.9, 10.3, 65.7, 93.0],
        'CMTN': [3.2, 327.5, 12.8, 30.8, 98.9],
        'STN': [1.7, 60.3, 11.3, 25.7, 92.9],
        'ETN': [1.8, 411.4, 77.7, 26.4, 74.4],
        'ATN': [1.5, 239.0, 9.0, 15.0, 98.5]
    }

    xs, ys, legends = [], [], []
    for key, value in data.items():
        legends.append(key)
        xs.append(value[-2])
        ys.append(value[-1])
    styles = get_styles(len(xs), lines=[''])
    plot(xs=xs, ys=ys, xlabel='Training time (min)', ylabel='Acc (%)', legends=legends, styles=styles, loc='lower right',
         markersize=8, filename='acc-time.png')
