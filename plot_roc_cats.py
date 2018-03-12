import h5py
from matplotlib import pyplot as plt
import numpy as np
import argparse
import src.utilities as U
import os
import json

plt.rcParams['figure.figsize'] = 5, 5
#use true type fonts only
plt.rcParams['pdf.fonttype'] = 42

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='file to load and plot')
    parser.add_argument('--batchmode', '-b',
                        help='run in batch mode', action='store_true')
    parser.add_argument('save', action='store_true',
                        help='whether to save the plots to disk')

    args = parser.parse_args()

    db = h5py.File(args.filename, 'r')

    plt_save_prefix = os.path.join(
        'plots/', args.filename.split('/')[-1].split('.')[0])
    #deal with the example images
    ims = db['example_imgs'].value
    ims = U.imagenet_deprocess(ims)
    #these are a little bit confusing; only show some images

    ims = ims[:224 * 5, :, :]
    #basic plot, showing the generated adversarial images
    plt.figure(1)
    plt.imshow(ims)
    plt.axis('off')
    plt.savefig(plt_save_prefix + 'example_imgs.png')
    #
    plt.figure(2)
    imsdiff = ims - np.concatenate([ims[:, :224, :]
                                    for _ in range(ims.shape[1] // 224)], axis=1)
    imsdiff[:, :224, :] = ims[:, :224, :]
    plt.imshow(imsdiff)
    plt.axis('off')
    plt.savefig(plt_save_prefix + 'example_ims_diff.png')

    plt.figure(3)
    # plot the ROC curves
    attack_info = db.get('attack')
    if attack_info is None:
        print('unknown attack')
    else:
        attack_spec = json.loads(attack_info.value)
        print(attack_spec['method'], attack_spec['eps'], attack_spec['ord'], sep=',') 
    print(',AUC entropy', 'AUC bald', 'AUC entropy (succ)', 'AUC bald (succ)', 'Accuracy: ', sep=',')
    print(attack_spec)
    for m in [k for k in db.keys() if 'Model' in k]:
        print(m, db[m]['entropy_AUC'].value, db[m]['bald_AUC'].value, db[m]['entropy_AUC_succ'].value, db[m]['bald_AUC_succ'].value, db[m]['adv_accuracy'].value, sep=',')

        plt.plot(db[m]['entropy_fpr'],
                 db[m]['entropy_tpr'],
                 label='Entropy {}'.format(m))

        plt.plot(db[m]['entropy_fpr_succ'],
                 db[m]['entropy_tpr_succ'],
                 label='Entropy {} (succ) '.format(m))

        if 'Deterministic' in m:
            continue

        plt.plot(db[m]['bald_fpr'],
                 db[m]['bald_tpr'],
                 label='Mutual Information {}'.format(m))

        plt.plot(db[m]['bald_fpr_succ'],
                 db[m]['bald_tpr_succ'],
                 label='Mutual Information {} (succ) '.format(m))

        plt.plot(np.linspace(0, 1, 10), np.linspace(
            0, 1, 10), color='k', linestyle='--')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    plt.savefig(plt_save_prefix + 'roc_curves.png')

    plt.figure(4)
    # plot the PR curves
    plt.plot(db['Deterministic Model']['entropy_rec'],
             db['Deterministic Model']['entropy_prec'],
             label='Entropy (deterministic model)')

    plt.plot(db['MC Model']['entropy_rec'],
             db['MC Model']['entropy_prec'],
             label='Entropy (MC model)')

    plt.plot(db['MC Model']['bald_rec'],
             db['MC Model']['bald_prec'],
             label='Mutual Information (MC model)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()

    plt.savefig(plt_save_prefix + 'precision_recall_curves.png')

    if not args.batchmode:
        plt.show()
    db.close()
