import matplotlib.pyplot as plt
import numpy as np


def generate_parallel_lines(slope, sigma, width, lines_count, points_count):
    biases = np.arange(0, width, width / lines_count)
    slopes = np.random.normal(slope, sigma, size=(lines_count, 1))
    xs = np.random.uniform(0, 1, size=(lines_count, points_count))

    points = xs * slopes

    noise = np.random.normal(0, 0.01, (lines_count, points_count))
    points = np.add(points, noise)

    for b, p, s in zip(biases, points, slopes):
        p += b

    plt.scatter(xs, points)
    plt.show()

    return xs, points, slopes

