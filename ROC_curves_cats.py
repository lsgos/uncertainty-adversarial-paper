import argparse
import h5py
import json

import numpy as np
from keras import backend as K
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from keras.utils import to_categorical
import src.utilities as U
from cleverhans import attacks
from cleverhans.model import CallableModelWrapper

from cats_and_dogs import H5PATH, define_model_resnet

"""
This script calculates the ROC for various models for the basic iterative method.
TODO: use CW attack? but this has a non-straightforward generalisation...
"""


def load_model(deterministic=False, name='save/cats_dogs_rn50_w_run.h5'):
    lp = not deterministic
    K.set_learning_phase(lp)
    model = define_model_resnet()

    model.load_weights(name)
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model


def make_random_targets(y, n_classes=10):
    """
    Return one hot vectors that differ from the labels in y
    """
    labels = y.argmax(axis=1)
    new = (labels + np.random.randint(1, n_classes - 1)) % n_classes
    return to_categorical(new, num_classes=n_classes)


def get_models(n_mc=10):
    models = []
    model = load_model(deterministic=True)
    models.append(('Deterministic Model', model))

    model = load_model(deterministic=False)
    input_tensor = model.input
    mc_model = U.MCModel(model, input_tensor, n_mc=n_mc)
    models.append(('MC Model', mc_model))

    return models


def batch_gen(array, batch_size=256):
    N = array.shape[0]
    n_batches = N // batch_size + (N % batch_size != 0)
    bs = batch_size
    return (array[i * bs:(i + 1) * bs] for i in range(n_batches))


def batch_eval(tensor, input_t, x, batch_size=256, verbose=False):
    bg = batch_gen(x, batch_size=batch_size)
    res = []
    for i, b in enumerate(bg):
        res.append(tensor.eval(session=K.get_session(), feed_dict={input_t: b}))
        if verbose:
            print(verbose, 'iteration: ', i)
    return np.concatenate(res, axis=0)


def create_adv_examples(model, input_t, x_to_adv, attack_dict):
    """
    This fn may seem bizarre and pointless, but the point of it is to
    enable the entire attack to be specified as a dict from the command line without
    editing this script, which is convenient for storing the settings used for an attack
    """
    if attack_dict['method'] == 'fgm':
        attack = attacks.FastGradientMethod(model, sess=K.get_session(), back='tf')
    elif attack_dict['method'] == 'bim':
        attack = attacks.BasicIterativeMethod(model, sess=K.get_session(), back='tf')
    elif attack_dict['method'] == 'mim':
        attack = attacks.MomentumIterativeMethod(model, sess=K.get_session(), back='tf')
    else:
        assert False, 'Current attack needs to be added to the create attack fn'
    adv_tensor = attack.generate(input_t, **{k: a for k, a in attack_dict.items() if
                                             k != 'method'})  # 'method' key for this fn use
    x_adv = batch_eval(adv_tensor, input_t, x_to_adv, batch_size=args.batch_size, verbose="Generating adv examples")
    return x_adv


def run(x_real,
        x_real_labels,
        x_to_adv,
        x_to_adv_labels,
        x_adv_labels,
        x_plus_noise,
        x_plus_noise_labels,
        x_advs_plot,
        attack_params,
        adv_save_num=15,
        fname='rcc_results_{}',
        batch_size=5,
        N_data=200):
    dists_ls = []

    fpr_entropies = []
    tpr_entropies = []

    fpr_balds = []
    tpr_balds = []

    prec_entropies = []
    rec_entropies = []

    prec_balds = []
    rec_balds = []

    AUC_entropies = []
    AUC_balds = []

    AP_entropies = []
    AP_balds = []
    # records on successful values
    fpr_entropies_succ = []
    tpr_entropies_succ = []

    fpr_balds_succ = []
    tpr_balds_succ = []

    prec_entropies_succ = []
    rec_entropies_succ = []

    prec_balds_succ = []
    rec_balds_succ = []

    AUC_entropies_succ = []
    AUC_balds_succ = []

    AP_entropies_succ = []
    AP_balds_succ = []

    accs = []
    modelnames = []
    for i, (name, m) in enumerate(models_to_eval):
        modelnames.append(name)

        input_t = K.placeholder(shape=(None, 224, 224, 3))
        wrap = CallableModelWrapper(m, 'probs')
        x_adv = create_adv_examples(wrap, input_t, x_to_adv, attack_params)

        # check the examples are really adversarial
        preds = np.concatenate([m.predict(x).argmax(axis=1) for x in batch_gen(x_adv, batch_size=args.batch_size)],
                               axis=0)
        acc = np.mean(np.equal(preds, x_to_adv_labels.argmax(axis=1)))
        print("Accuracy on adv examples:", acc)
        accs.append(acc)

        succ_adv_inds = np.logical_not(
            np.equal(preds, x_to_adv_labels.argmax(axis=1)))  # seperate out succesful adv examples

        dists = U.batch_L_norm_distances(x_to_adv, x_adv, ord=2)
        noise = np.random.random(size=x_plus_noise.shape)
        noise /= (dists * np.linalg.norm(noise.reshape(x_plus_noise.shape[0], -1), axis=1))[:, None, None, None]
        x_plus_noise += noise
        x_plus_noise = np.clip(x_plus_noise, 0, 1)
        x_synth = np.concatenate([x_real, x_adv, x_plus_noise])
        y_synth = np.array(x_real_labels + x_adv_labels + x_plus_noise_labels)
        dists_ls.append(dists)
        succ_adv_inds = np.concatenate(
            [np.ones(len(x_real_labels)), succ_adv_inds, np.ones(len(x_plus_noise_labels))]).astype(np.bool)
        # save the adverserial examples to plot
        x_advs_plot = x_advs_plot + [U.tile_images([x_adv[i] for i in range(adv_save_num)], horizontal=False)]

        batches = U.batches_generator(x_synth, y_synth, batch_size=batch_size)
        # get the entropy and bald on this task
        try:
            # we can now clean up the adv tensor
            del input_t
            del adv_tensor
        except:
            pass  # if these aren't defined, ignore

        entropy = []
        bald = []
        for j, (bx, by) in enumerate(batches):
            print('Evaluating entropy/bald: batch ', j)
            if hasattr(m, 'get_results'):
                _, e, b = m.get_results(bx)
            else:
                res = m.predict(bx)
                e = np.sum(- res * np.log(res + 1e-6), axis=1)
                b = np.zeros(e.shape)  # undefined
            entropy.append(e)
            bald.append(b)

        entropy = np.concatenate(entropy, axis=0)
        bald = np.concatenate(bald, axis=0)

        fpr_entropy, tpr_entropy, _ = roc_curve(y_synth, entropy, pos_label=1)
        fpr_bald, tpr_bald, _ = roc_curve(y_synth, bald, pos_label=1)

        prec_entr, rec_entr, _ = precision_recall_curve(y_synth, entropy, pos_label=1)
        prec_bald, rec_bald, _ = precision_recall_curve(y_synth, bald, pos_label=1)

        AUC_entropy = roc_auc_score(y_synth, entropy)
        AUC_bald = roc_auc_score(y_synth, bald)

        AP_entropy = average_precision_score(y_synth, entropy)
        AP_bald = average_precision_score(y_synth, bald)

        fpr_entropies.append(fpr_entropy)
        tpr_entropies.append(tpr_entropy)

        prec_entropies.append(prec_entr)
        rec_entropies.append(rec_entr)

        prec_balds.append(prec_bald)
        rec_balds.append(rec_bald)

        fpr_balds.append(fpr_bald)
        tpr_balds.append(tpr_bald)

        AUC_entropies.append(AUC_entropy)
        AUC_balds.append(AUC_bald)

        AP_entropies.append(AP_entropy)
        AP_balds.append(AP_bald)

        # record stats on successful adv examples only
        y_synth = y_synth[succ_adv_inds]
        entropy = entropy[succ_adv_inds]
        bald = bald[succ_adv_inds]

        fpr_entropy, tpr_entropy, _ = roc_curve(y_synth, entropy, pos_label=1)
        fpr_bald, tpr_bald, _ = roc_curve(y_synth, bald, pos_label=1)

        prec_entr, rec_entr, _ = precision_recall_curve(y_synth, entropy, pos_label=1)
        prec_bald, rec_bald, _ = precision_recall_curve(y_synth, bald, pos_label=1)

        AUC_entropy = roc_auc_score(y_synth, entropy)
        AUC_bald = roc_auc_score(y_synth, bald)

        AP_entropy = average_precision_score(y_synth, entropy)
        AP_bald = average_precision_score(y_synth, bald)

        fpr_entropies_succ.append(fpr_entropy)
        tpr_entropies_succ.append(tpr_entropy)

        prec_entropies_succ.append(prec_entr)
        rec_entropies_succ.append(rec_entr)

        prec_balds_succ.append(prec_bald)
        rec_balds_succ.append(rec_bald)

        fpr_balds_succ.append(fpr_bald)
        tpr_balds_succ.append(tpr_bald)

        AUC_entropies_succ.append(AUC_entropy)
        AUC_balds_succ.append(AUC_bald)

        AP_entropies_succ.append(AP_entropy)
        AP_balds_succ.append(AP_bald)

    fname = U.gen_save_name(fname.format(attack_params["method"]))

    with h5py.File(fname, 'w') as f:
        # record attack information into the results
        f.create_dataset('attack', data=json.dumps(attack_params))
        f.create_dataset('dists', data=np.array(dists_ls))
        f.create_dataset('N_data', data=N_data)
        for i, name in enumerate(modelnames):
            g = f.create_group(name)
            g.create_dataset('entropy_fpr', data=fpr_entropies[i])
            g.create_dataset('entropy_tpr', data=tpr_entropies[i])
            g.create_dataset('bald_fpr', data=fpr_balds[i])
            g.create_dataset('bald_tpr', data=tpr_balds[i])
            g.create_dataset('entropy_prec', data=prec_entropies[i])
            g.create_dataset('entropy_rec', data=rec_entropies[i])
            g.create_dataset('bald_prec', data=prec_balds[i])
            g.create_dataset('bald_rec', data=rec_balds[i])
            g.create_dataset('entropy_AUC', data=AUC_entropies[i])
            g.create_dataset('bald_AUC', data=AUC_balds[i])
            g.create_dataset('entropy_AP', data=AP_entropies[i])
            g.create_dataset('bald_AP', data=AP_balds[i])

            g.create_dataset('entropy_fpr_succ', data=fpr_entropies_succ[i])
            g.create_dataset('entropy_tpr_succ', data=tpr_entropies_succ[i])
            g.create_dataset('bald_fpr_succ', data=fpr_balds_succ[i])
            g.create_dataset('bald_tpr_succ', data=tpr_balds_succ[i])
            g.create_dataset('entropy_prec_succ', data=prec_entropies_succ[i])
            g.create_dataset('entropy_rec_succ', data=rec_entropies_succ[i])
            g.create_dataset('bald_prec_succ', data=prec_balds_succ[i])
            g.create_dataset('bald_rec_succ', data=rec_balds_succ[i])
            g.create_dataset('entropy_AUC_succ', data=AUC_entropies_succ[i])
            g.create_dataset('bald_AUC_succ', data=AUC_balds_succ[i])
            g.create_dataset('entropy_AP_succ', data=AP_entropies_succ[i])
            g.create_dataset('bald_AP_succ', data=AP_balds_succ[i])

            g.create_dataset('adv_accuracy', data=accs[i])

        f.create_dataset('example_imgs', data=np.concatenate(x_advs_plot, axis=1))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--N_data', type=int, default=100, help="Number of examples \
        of adverserial and non-adverserial examples to use.")
    parser.add_argument('--N_mc', type=int, default=20, help="number of mc passes")
    parser.add_argument('--batch_size', type=int, default=5, help='Batch size to use')

    args = parser.parse_args()

    #  imagenet min/max after preprocessing

    SYNTH_DATA_SIZE = args.N_data

    print('Loading data...')
    h5database = h5py.File(H5PATH, 'r')
    x_test = h5database['test']['X'].value
    y_test = h5database['test']['Y'].value
    h5database.close()

    # load the pre-trained models 
    models_to_eval = get_models(n_mc=args.N_mc)

    # create a synthetic training set at various epsilons, 
    # and evaluate the ROC curves on it. Combine adversarial and random pertubations

    x_real = x_test[np.random.randint(x_test.shape[0], size=SYNTH_DATA_SIZE)]
    to_adv_inds = np.random.randint(x_test.shape[0], size=SYNTH_DATA_SIZE)
    x_to_adv = x_test[to_adv_inds]
    x_to_adv_labels = y_test[to_adv_inds]
    x_plus_noise = x_test[np.random.randint(x_test.shape[0], size=SYNTH_DATA_SIZE)]

    adv_save_num = 15 if x_to_adv.shape[0] >= 15 else x_to_adv.shape[0]

    x_advs_plot = [U.tile_images([x_to_adv[i] for i in range(adv_save_num)], horizontal=False)]

    # label zero for non adverserial input
    x_real_labels = [0 for _ in range(SYNTH_DATA_SIZE)]
    x_plus_noise_labels = [0 for _ in range(SYNTH_DATA_SIZE)]
    x_adv_labels = [1 for _ in range(SYNTH_DATA_SIZE)]

    attack_params = [
        {
            "method": "fgm",
            "eps": 5,
            "clip_min": -103.939,
            "clip_max": 131.32,
            "ord": np.inf,
        },
        {
            "method": "fgm",
            "eps": 10,
            "clip_min": -103.939,
            "clip_max": 131.32,
            "ord": np.inf,
        },
        {
            "method": "bim",
            "eps": 5,
            "clip_min": -103.939,
            "clip_max": 131.32,
            "ord": np.inf,
            "nb_iter": 10,
            "eps_iter": 0.5
        },
        {
            "method": "bim",
            "eps": 10,
            "eps_iter": 1.2,
            "clip_min": -103.939,
            "clip_max": 131.32,
            "ord": np.inf,
        },
        {
            "method": "mim",
            "eps": 5,
            "clip_min": -103.939,
            "clip_max": 131.32,
            "ord": np.inf,
            "nb_iter": 10,
            "eps_iter": 0.5
        },
        {
            "method": "mim",
            "eps": 10,
            "eps_iter": 1.2,
            "clip_min": -103.939,
            "clip_max": 131.32,
            "ord": np.inf,
        }]
    for attack_spec in attack_params:
        print(attack_spec)
        run(x_real,
            x_real_labels,
            x_to_adv,
            x_to_adv_labels,
            x_adv_labels,
            x_plus_noise,
            x_plus_noise_labels,
            x_advs_plot,
            attack_spec,
            adv_save_num=adv_save_num,
            fname='my_roc_curves_{}.h5',
            batch_size=args.batch_size,
            N_data=args.N_data)
