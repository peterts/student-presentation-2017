import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

# Parameters
max_x_value = 10
n_samples = 100
learning_rate = 0.05
w_real = np.asarray([1, 1 / 2])

# Initialize data sample
constant = np.ones(n_samples)
x = np.random.sample(n_samples) * max_x_value
x = np.vstack([constant, x]).T
eps = np.random.normal(0.0, 1.0, n_samples)
y_real = np.sum(w_real * x, axis=1) + eps
plt.scatter(x[:, 1], y_real)
plt.show()


# Gradient descent
def gradient_descent(args=None):
    # Initialize the weights to some random numbers
    w_pred = np.random.random(2)
    for i in range(1000):
        # Perform the weight update
        y_pred = np.sum(w_pred * x, axis=1)
        gradient = np.dot(y_pred - y_real, x) / n_samples
        w_pred = w_pred - learning_rate * gradient
        print(w_pred)

        # Return the new prediction
        yield y_pred


# Update the plot
def update_line(y_pred):
    line.set_data(x[:, 1], y_pred)
    return line,


# Plot figure settings
fig = plt.figure()
line, = plt.plot([], [], 'r', linewidth=2)
plt.scatter(x[:, 1], y_real)

# Run gradient descent and animation
ani = animation.FuncAnimation(fig, update_line, gradient_descent)
plt.show()

