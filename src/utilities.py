import tensorflow as tf
import keras
import numpy as np
import os
from keras import backend as K
from keras.datasets import mnist
import itertools as itr
from functools import reduce
import operator
import re
from PIL import Image
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential

def make_grid(im_batch, rect):
    """
    Concatenate a batch of samples into an n by n image
    """
    h, w = rect
    ims = [im.squeeze() for im in im_batch]

    ims = [ims[i* w:(i+1)*w] for i in range(h)]

    ims = [np.concatenate(xs, axis=0) for xs in ims]

    ims = np.concatenate(ims, axis=1)
    return ims



def imagenet_deprocess(x, mode='caffe'):
    """
    Reverses the transformation that imagenet does on the images. This
    is the inverse function of the keras utility which can be found at
    https://github.com/keras-team/keras/blob/master/keras/applications/i
    magenet_utils.py 
    """
    data_format = K.image_data_format()

    if mode == 'tf':
        x /= 127.5
        x -= 1.
        return x

    if mode == 'torch':
        x /= 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [103.939, 116.779, 123.68]
        std = None

    # Zero-center by mean pixel
    if data_format == 'channels_first':
        if x.ndim == 3:
            x[0, :, :] += mean[0]
            x[1, :, :] += mean[1]
            x[2, :, :] += mean[2]
            if std is not None:
                x[0, :, :] *= std[0]
                x[1, :, :] *= std[1]
                x[2, :, :] *= std[2]
        else:
            x[:, 0, :, :] += mean[0]
            x[:, 1, :, :] += mean[1]
            x[:, 2, :, :] += mean[2]
            if std is not None:
                x[:, 0, :, :] *= std[0]
                x[:, 1, :, :] *= std[1]
                x[:, 2, :, :] *= std[2]
    else:
        x[..., 0] += mean[0]
        x[..., 1] += mean[1]
        x[..., 2] += mean[2]
        if std is not None:
            x[..., 0] *= std[0]
            x[..., 1] *= std[1]
            x[..., 2] *= std[2]
    #undo the channel switch to go back to RGB
    if data_format == 'channels_first':
        # 'BGR'->'RGB'
        if x.ndim == 3:
            x = x[::-1, ...]
        else:
            x = x[:, ::-1, ...]
    else:
        # 'BGR'->'RGB'
        x = x[..., ::-1]
    x = np.clip(x, 0, 225)
    return x.astype(np.uint8)

class MCEnsembleWrapper:
    """
    This class wraps a list of models all of which are mc models
    """
    def __init__(self, modellist, n_mc):
        self.ms = modellist
        self.n_mc = n_mc
    def predict(self, X):
        mc_preds = np.concatenate( [np.stack([m.predict(X) for _ in range(self.n_mc)])
                                    for m in self.ms], axis=0)
        return mc_preds.mean(axis=0)
    def get_results(self, X):
        mc_preds = np.concatenate( [np.stack([m.predict(X) for _ in range(self.n_mc)])
                                    for m in self.ms], axis=0)
        preds = mc_preds.mean(axis=0)
        ent = - 1 *np.sum(preds * np.log(preds + 1e-10), axis=-1)
        bald = ent - np.mean( - 1 * np.sum(mc_preds * np.log(mc_preds + 1e-10), axis=-1), axis=0)
        return preds, ent, bald
    def __call__(self, X):
        """
        Returns the mean prediction of the entire ensemble as a keras tensor to allow differentiation
        """
        return K.mean(
            K.stack(
                [K.mean(mc_dropout_preds(m, X, n_mc=self.n_mc), axis=0)
                     for m in self.ms]
            ),
            axis=0)
def crop_center_or_reshape(im, size):
     
    tw, th = size

    im.thumbnail((int(tw * 1.5), int(th + 1.5)) )#heuristic 
    iw,ih = im.size

    left = np.ceil((iw - tw) / 2)
    right = iw - np.floor((iw - tw) / 2)
    top = np.ceil((ih - th) / 2)
    bottom = ih - np.floor((ih - th) / 2)
    im = im.crop((left, top, right, bottom))
    if im.size != size:
        raise RuntimeError
    return im
    
def load_jpgs(path, size=(224,224)):
    """
    Load all jpgs on a path into a numpy array, resizing to a given image size
    """
    fnames = os.listdir(path)
    imgs = []
    for f in fnames:
        if not re.match('.*\.(jpg|jpeg|JPEG|JPG)', f):
            continue
        try:
            im = Image.open(os.path.join(path,f))
        except OSError:
            continue #ignore corrupt files
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im = crop_center_or_reshape(im, size)
        img = np.asarray(im)
        #img = tf.image.resize_image_with_crop_or_pad(x, size[0], size[1]).eval(session=K.get_session())
            
        imgs.append(img)
        
    return np.array(imgs)

def calc_nn_dist(X, y, ord=2):
    """
    Calculate the distance between y and it's nearest neighbour in x
    X is N x d, y is d.
    """

    #if the inputs are not 2d, flatten the remaining directions
    
    X = X.reshape(X.shape[0], -1)
    y = y.flatten()
    return np.linalg.norm(X - y, axis=1, ord=ord).min()


class MCModel:
    def __init__(self,model, input_tensor, n_mc):
        self.model = model
        self.input = input_tensor
        self.n_mc=n_mc
        self.mc_preds_t = mc_dropout_preds(self.model, self.input, n_mc=n_mc)
        self.predictive_entropy_t = predictive_entropy(self.mc_preds_t)
        self.expected_entropy_t   = expected_entropy(self.mc_preds_t)
        self.bald_t = self.predictive_entropy_t - self.expected_entropy_t

    def get_results(self,x):
        f = K.function([self.input],
                       [K.mean(self.mc_preds_t, axis=0),
                        self.predictive_entropy_t, self.bald_t])
        return f([x])
    def get_mc_preds(self, x):
        f = K.function([self.input], [self.mc_preds_t])
        return f([x])[0]

    def predict(self, x):
        return self.get_results(x)[0]
    def __call__(self, x):
        return K.mean(mc_dropout_preds(self.model, x, n_mc = self.n_mc), axis=0)

def gen_save_name(basename: str):
    """
    Generate a unique name for a saved file to avoid overwrites.
    """
    fname, suffix = basename.split('.') #just assume this is true.
    qualifier = 1
    unique_fname = fname
    while(os.path.exists(unique_fname + '.' + suffix)):
        unique_fname = fname + '_{}'.format(qualifier)
        qualifier += 1
    return unique_fname + '.' + suffix

def create_unique_folder(basepath: str):
    """
    Create a unique variation on basepath by appending 1,2,3,...
    """
    num = 0
    path = basepath + repr(num)
    while(os.path.exists(path)):
        num += 1
        path = basepath + repr(num)
    #path is now a unquie name
    os.mkdir(path)
    return path

def batch_L_norm_distances(X: np.array, Y: np.array, ord=2) -> np.array:
    """
    Takes 2 arrays of N x d examples and calculates the p-norm between
    them. Result is dimension N. If the inputs are N x h x w etc, they
    are first flattened to be N x d
    """
    assert X.shape == Y.shape, "X and Y must have the same dimensions"
    N = X.shape[0]
    rest = X.shape[1:]
    d = reduce(operator.mul, rest, 1)  # product of leftover dimensions

    x = X.reshape(N, d)
    y = Y.reshape(N, d)

    if ord == 2:
        return np.sum((x - y) ** 2, axis=1)
    elif ord == 1:
        return np.sum(np.abs(x - y), axis=1)
    elif ord == 0:
        return np.isclose(x, y).astype(np.float).sum(axis=1)
        # return the number of entries in x that differ from y.
        # Use a tolerance to allow numerical precision errors.
    elif ord == np.inf:
        return np.max(np.abs(x - y), axis=1)
    else:
        raise NotImplementedError(
            "Norms other than 0, 1, 2, inf not implemented")


def tile_images(imlist: [np.array], horizontal=True) -> np.array:
    """
    Takes a list of images and tiles them into a single image for plotting
    purposes.
    """
    ax = 1 if horizontal else 0
    tile = np.concatenate([x.squeeze() for x in imlist], axis=ax)
    return tile


def get_mnist():
    """
    Return the mnist data, scaled to [0,1].
    """
    num_classes = 10
    img_rows, img_cols = 28, 28
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def mc_dropout_preds(model, x: tf.Tensor, n_mc: int) -> tf.Tensor:
    """
    Take a model, and a tensor of size batch_size x n_classes and return the
    result of doing n_mc stochastic forward passes as a n_mc x batch_size x
    n_classes tensor. This assumes the model has some VI layers like dropout or
    whatever, and that the model has been loaded with
    keras.backend.set_learning_phase(True). Also note that this takes and
    returns keras tensors, not arrays.
    """
    # tile x n_mc times and predict in a batch
    xs = K.stack(list(itr.repeat(x, n_mc)))
    mc_preds = K.map_fn(model, xs)  # [n_mc x batch_size x n_classes]
    return mc_preds


def entropy(X: tf.Tensor) -> tf.Tensor:
    return K.sum(- X * K.log(K.clip(X, 1e-6, 1)), axis=-1)


def expected_entropy(mc_preds: tf.Tensor) -> tf.Tensor:
    """
    Take a tensor mc_preds [n_mc x batch_size x n_classes] and return the
    mean entropy of the predictive distribution across the MC samples.
    """

    return K.mean(entropy(mc_preds), axis=0)  # batch_size


def predictive_entropy(mc_preds: tf.Tensor) -> tf.Tensor:
    """
    Take a tensor mc_preds [n_mc x batch_size x n_classes] and return the
    entropy of the mean predictive distribution across the MC samples.
    """
    return entropy(K.mean(mc_preds, axis=0))


def BALD(mc_preds: tf.Tensor) -> tf.Tensor:
    """
    Calculate the BALD (Bayesian Active Learning by Disagreement) of a model;
    the difference between the mean of the entropy and the entropy of the mean
    of the predicted distribution on the n_mc x batch_size x n_classes tensor
    mc_preds. In the paper, this is referred to simply as the MI.
    """
    BALD = predictive_entropy(mc_preds) - expected_entropy(mc_preds)
    return BALD


def batches_generator(x: np.array, y: np.array, batch_size=100):
    """
    Yield a generator of batches to iterate easily through a dataset.
    """
    # todo; maybe add the ability to shuffle?
    N = x.shape[0]
    n_batches = N // batch_size + (N % batch_size != 0)
    for i in range(n_batches):
        lo = i * batch_size
        hi = (i + 1) * batch_size
        yield x[lo:hi], y[lo:hi]


def batch_eval(k_function, batch_iterable):
    """
    eval a keras function across a list, hiding the fact that keras requires
    you to pass a list to everything for some reason.
    """
    return [k_function([bx]) for bx in batch_iterable]
