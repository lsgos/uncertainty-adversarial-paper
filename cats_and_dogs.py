import keras
import numpy as np
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import Dropout, Dense
import src.utilities as U
import os
import h5py
from keras import backend as K

H5PATH = '/data-local/lsgs/cats_dogs.h5'


def load_or_create_dataset():
    if not os.path.exists(H5PATH):
        cats = U.load_jpgs('/data-local/lsgs/PetImages/Cat')
        catlabel = np.zeros(cats.shape[0])

        dogs = U.load_jpgs('/data-local/lsgs/PetImages/Dog')
        doglabel = np.ones(dogs.shape[0])

        data = np.concatenate([cats, dogs])
        labels = np.concatenate([catlabel, doglabel])

        inds = np.random.permutation(data.shape[0])

        X = preprocess_input(data.astype(np.float))
        Y = keras.utils.to_categorical(labels)

        # shuffle data
        X = X[inds]
        Y = Y[inds]

        N = X.shape[0]
        split = int(0.8 * N)

        X_train = X[:split]
        Y_train = Y[:split]

        X_test = X[split:]
        Y_test = Y[split:]

        # write to database file to avoid this crap later
        with h5py.File(H5PATH, 'w') as f:
            tr = f.create_group('train')
            te = f.create_group('test')
            tr.create_dataset('X', data=X_train)
            tr.create_dataset('Y', data=Y_train)

            te.create_dataset('X', data=X_test)
            te.create_dataset('Y', data=Y_test)
        return X_train, Y_train, X_test, Y_test
    else:
        with h5py.File(H5PATH, 'r') as f:
            X_train = f['train']['X'].value
            Y_train = f['train']['Y'].value

            X_test = f['test']['X'].value
            Y_test = f['test']['Y'].value
        return X_train, Y_train, X_test, Y_test


def define_model_resnet():
    K.set_learning_phase(True)
    rn50 = ResNet50(weights='imagenet', include_top='False')
    a = Dropout(rate=0.5)(rn50.output)
    a = Dense(2, activation='softmax')(a)

    model = keras.models.Model(inputs=rn50.input, outputs=a)

    # freeze resnet layers
    for layer in rn50.layers:
        layer.trainable = False
    return model


if __name__ == '__main__':
    model = define_model_resnet()
    wname = 'save/cats_dogs_rn50_w_run.h5'

    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'], optimizer='adam')
    with h5py.File(H5PATH, 'r') as f:
        x_tr = f['train']['X']
        y_tr = f['train']['Y']

        x_te = f['test']['X']
        y_te = f['test']['Y']
        model.fit(x_tr, y_tr, epochs=15, validation_data=(x_te, y_te), shuffle='batch')
        name = U.gen_save_name(wname)
        model.save_weights(name)
