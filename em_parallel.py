from generate_parallel_lines import generate_parallel_lines
from em import e_step, m_step

import numpy as np

xs, lines, slopes = generate_parallel_lines(1, 0.1, 10, 5, 10)

all_x = xs.flatten()
all_y = np.expand_dims(lines.flatten(), -1)

all_x = np.hstack((np.expand_dims(all_x, -1), np.ones((len(all_y), 1))))

probs = np.ones((5, 1)) / 5

weights = np.random.normal(0, 10, size=(5, 2, 1))

print(weights)
for _ in range(100):
    p = e_step(probs, weights, all_x, all_y)
    probs, weights = m_step(p, weights, all_x, all_y)

print(probs)
print(weights)