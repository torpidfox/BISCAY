import numpy as np

def generate_function_data(f,
                  size=500,
                  noise=.1):
    if size == 0:
        return [], []

    x = np.random.uniform(0, 1, size=size)

    if noise > 0:
        noise = np.random.normal(scale=noise, size=size)
        y = f(x) + noise
    else:
        y = f(x)

    return x, y

def generate_regression_data(w, b, size):
    f = lambda x: w * x + b

    return generate_function_data(f, size)

def generate_background_noise(low_y, high_y, size):
    y = np.random.uniform(low_y, high_y, size=(size, 1))
    x = np.random.uniform(0, 1, size=(size, 1))

    return x.flatten(), y.flatten()