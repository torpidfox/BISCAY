import numpy as np
import matplotlib.pyplot as plt

def logsumexp(x):
    #print(x)
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

# probs: n x m matrix each row - probability of classes for sample
def m_step(probs, weights, x, y):
    x_sq = np.dot(np.multiply(x.T, probs[:, 0]), x)
    inv = np.linalg.inv(x_sq)
    #inv = 1/(x_sq + 1e-100) #for no bias case
    rest = np.dot(
        np.multiply(x.T, probs[:, 0]), y)

    beta = np.dot(inv, rest)

    pi = np.sum(probs.T, axis=1) / len(y)


    return np.expand_dims(pi, axis=1), beta

def e_step(pi, weights, left, right, x, y):
    power = np.power(((y - np.dot(x, weights)) ** 2) / 0.01, pi[0])

    probs1 = np.exp(-power)
    probs2 = np.full((len(x), 1), 1.0/(right - left))

    pdf = np.hstack((probs1, probs2))

    class_weights = []

    for sample in pdf:
        class_weights.append(sample / np.sum(sample))

    #pdf = np.asarray(pdf)
    #class_weights = pdf

    #class_weights = np.exp(pdf - logsumexp(pdf))
    #class_weights /= np.sum(class_weights.T, axis=1)

    return np.asarray(class_weights)


count = 100
x1 = np.random.uniform(0, 1, count)
y1 = 1 * x1 + 1 + np.random.normal(0, 0.1, count)
x2 = np.random.uniform(0, 2, count)
y2 = np.random.uniform(0, 2, count)

plt.scatter(x1, y1)
plt.scatter(x2, y2)


x1 = np.hstack( (np.expand_dims(x1, -1), np.ones((len(x1), 1))) )
x2 = np.hstack( (np.expand_dims(x2, -1), np.ones((len(x1), 1))) )
all_x = np.concatenate((x1, x2))
all_y = np.expand_dims(np.concatenate((y1, y2)), -1)

probs = np.array([[0.7], [0.3]])
weights = np.array([[1], [1]])

for _ in range(100):
    p = e_step(probs, weights, 0, 2, all_x, all_y)
    probs, weights = m_step(p, weights,  all_x, all_y)
    print(weights)

print(probs)
print(weights)