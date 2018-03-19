import keras
from keras import backend as K
from keras.layers import Dense, Flatten, Lambda, Reshape
from keras.models import Model

import src.utilities as U

BATCH_SIZE = 128


def define_VAE(optim='adagrad', latent_dim=2):
    inputs = keras.layers.Input(shape=(28, 28, 1))
    x = Flatten()(inputs)
    enc_1 = Dense(400, activation='elu')(x)
    enc_2 = Dense(256, activation='elu')(enc_1)

    z_mu = Dense(latent_dim)(enc_2)
    z_logsigma = Dense(latent_dim)(enc_2)

    encoder = Model(inputs=inputs, outputs=z_mu)  # represent the latent space by the mean

    def sample_z(args):
        mu, logsigma = args
        return 0.5 * K.exp(logsigma / 2) * K.random_normal(shape=(K.shape(mu)[0], latent_dim)) + mu

    z = Lambda(sample_z, output_shape=(latent_dim,))([z_mu, z_logsigma])

    dec_input = keras.layers.Input(shape=(latent_dim,))
    dec_1 = Dense(256, activation='elu')(dec_input)
    dec_2 = Dense(400, activation='elu')(dec_1)
    dec_output = Dense(784, activation='sigmoid')(dec_2)

    dec_reshaped = Reshape((28, 28, 1))(dec_output)
    decoder = Model(inputs=dec_input, outputs=dec_reshaped)

    reconstruction = decoder(z)

    VAE = Model(inputs=inputs, outputs=reconstruction)

    def vae_loss(inputs, reconstruction):
        x = K.flatten(inputs)
        rec = K.flatten(reconstruction)
        x_ent = keras.metrics.binary_crossentropy(x, rec)
        kl_div = 0.5 * K.sum(K.exp(z_logsigma) + K.square(z_mu) - z_logsigma - 1, axis=-1)
        return 28 * 28 * x_ent + kl_div

    VAE.compile(optimizer=optim, loss=vae_loss)

    return VAE, encoder, decoder


if __name__ == '__main__':
    latent_dim = 2
    x_train, y_train, x_test, y_test = U.get_mnist()

    VAE, encoder, decoder = define_VAE(
        optim=keras.optimizers.Adam(),
        latent_dim=latent_dim,
    )

    VAE.fit(x_train, x_train,
            epochs=50,
            batch_size=BATCH_SIZE,
            validation_data=(x_test, x_test))

    encoder.save_weights('save/enc_weights_latent_dim_' + str(latent_dim) + '.h5')
    decoder.save_weights('save/dec_weights_latent_dim_' + str(latent_dim) + '.h5')
