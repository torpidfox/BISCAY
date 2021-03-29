import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from generate_parallel_lines import generate_parallel_lines

def logsumexp(x):
    #print(x)
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

# probs: n x m matrix each row - probability of classes for sample
def m_step(probs, weights, x, y):
    beta = []

    for i, w in enumerate(weights):
        x_sq = np.dot(np.multiply(x.T, probs[:, i]), x)
        inv = np.linalg.inv(x_sq)
        #inv = 1/(x_sq + 1e-100) #for no bias case
        rest = np.dot(
            np.multiply(x.T, probs[:, i]), y)

        beta.append(np.dot(inv, rest))

    pi = np.sum(probs.T, axis=1) / len(y)

    return np.expand_dims(pi, axis=1), np.asarray(beta)

def e_step(pi, weights, x, y):
    pdf = []

    for prob, w in zip(pi, weights):
        power = np.power(((y - np.dot(x, w)) ** 2) / 0.01, prob)
        #pdf.append(np.exp(-power))
        pdf.append(-power)



    pdf = np.asarray(pdf).T[0]
    class_weights = []

    for sample in pdf:
        class_weights.append(np.exp(sample - logsumexp(sample)))

    #pdf = np.asarray(pdf)
    #class_weights = pdf

    #class_weights = np.exp(pdf - logsumexp(pdf))
    #class_weights /= np.sum(class_weights.T, axis=1)

    return np.asarray(class_weights)


x1 = np.random.uniform(0, 10, 10)
y1 = 1 * x1 + 1 + np.random.normal(0, 1, 10)
x2 = np.random.uniform(0, 10, 10)
y2 = -1 * x2 - 1 + np.random.normal(0, 1, 10)


plt.scatter(x1, y1, label='5*x')
plt.scatter(x2, y2, label='-5*x')
plt.legend()
x1 = np.hstack( (np.expand_dims(x1, -1), np.ones((len(x1), 1))) )
x2 = np.hstack( (np.expand_dims(x2, -1), np.ones((len(x1), 1))) )
all_x = np.concatenate((x1, x2))
all_y = np.expand_dims(np.concatenate((y1, y2)), -1)

probs = np.array([[0.5], [0.5]])
weights = np.array([[[2], [0]], [[-2], [0]]])

print(all_x.shape, all_y.shape)

for _ in range(1000):
    p = e_step(probs, weights, all_x, all_y)
    probs, weights = m_step(p, weights, all_x, all_y)

print(probs)
print(weights)