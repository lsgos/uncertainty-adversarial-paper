import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

import src.utilities as U

from latent_plots import get_models, visualise_latent_space

plt.rcParams['figure.figsize'] = 8, 5
#use true type fonts only
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42 

if __name__ == '__main__':
    model, encoder, decoder = get_models()

    #move along a random line in latent space

    _,_,mnist, label = U.get_mnist()
    x1 = mnist[label.argmax(axis=1) == 6][200]
    x2 = mnist[label.argmax(axis=1) == 8][200]

    x_ims = np.stack([(1 - t) * x1 + t * x2 for t in np.linspace(0,1, 15)])

    x_preds, x_entropy, x_bald = model.get_results(x_ims)
 
    z_begin = encoder.predict(x1[None, :]).flatten()

    z_end   = encoder.predict(x2[None, :]).flatten()

    z_lin = np.stack([(1 - t) * z_begin  + t * z_end for t in np.linspace(0,1,15)])

    z_ims = decoder.predict(z_lin)

    z_preds, z_entropy, z_bald = model.get_results(z_ims)

   
    f = plt.figure()
    gs = gridspec.GridSpec(4, 1, height_ratios=[1,1,3,3])

    ax0 = plt.subplot(gs[0])
    ax0.set_axis_off()
    ax0.imshow(np.concatenate([im.squeeze() for im in z_ims], axis=1), extent=[-.5, z_ims.shape[0] + .5, 0, 1], cmap='gray_r')

    ax1 = plt.subplot(gs[1])
    ax1.set_axis_off()
    ax1.imshow(np.concatenate([im.squeeze() for im in x_ims], axis=1), extent=[-.5, x_ims.shape[0] + .5, 0, 1], cmap='gray_r')


    ax2 = plt.subplot(gs[2])
    ax2.plot(z_entropy, label='Latent Space', c ='r')
    ax2.plot(x_entropy, label='Image Space', c = 'b')
    ax2.legend()
    
    ax3 = plt.subplot(gs[3])
    ax3.plot(z_bald, label='Latent Space', c = 'r')
    ax3.plot(x_bald, label='Image Space', c = 'b')
    ax3.legend()

    plt.savefig('my-figure.png')
    plt.show()

