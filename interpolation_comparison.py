import numpy as np
from matplotlib import pyplot as plt

import src.utilities as U

from latent_plots import get_models


def create_interpolation_dataset(x, y, encoder, decoder, n=100):
    chosen = np.random.permutation(x.shape[0])

    x_1 = x[chosen][:n]
    y_1 = y[chosen][:n]

    # choose datapoints of different classes

    x_2 = []
    for l in y_1:
        feasible = x[y.argmax(axis=1) != l.argmax()]
        choice = np.random.randint(feasible.shape[0])
        x_2.append(feasible[choice])

    x_2 = np.array(x_2)

    # generate interpolations in image space

    ts = np.ones(n) * 0.5

    ts = ts.reshape(n, 1, 1, 1)
    pixel_interp = ts * x_1 + (1 - ts) * x_2

    z_1 = encoder.predict(x_1)
    z_2 = encoder.predict(x_2)

    ts = ts.reshape(n, 1)

    latent_interp = ts * z_1 + (1 - ts) * z_2

    latent_interp = decoder.predict(latent_interp)

    labels = np.concatenate([np.ones(n), np.zeros(n)])
    xs = np.concatenate([pixel_interp, latent_interp], axis=0)

    return xs, labels


if __name__ == '__main__':
    model, encoder, decoder = get_models()

    # move along a random line in latent space

    _, _, mnist, label = U.get_mnist()

    x, y = create_interpolation_dataset(mnist, label, encoder, decoder, n=3000)

    _, entropy, mi = model.get_results(x)

    f, ax = plt.subplots(2, 1)

    ax[0].hist(entropy[y == 0], color='r', alpha=0.5, label='Latent Space')
    ax[0].hist(entropy[y == 1], color='b', alpha=0.5, label='Pixel Space')
    ax[0].legend()

    ax[1].hist(mi[y == 0], color='r', alpha=0.5, label='Latent Space')
    ax[1].hist(mi[y == 1], color='b', alpha=0.5, label='Pixel Space')
    ax[1].legend()

    plt.savefig('path-to-my-figure.png')
    plt.show()
