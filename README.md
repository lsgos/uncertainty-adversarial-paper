This repository provides code needed to reproduce the experiments in my
'[Understanding Measures of Uncertainty for Adversarial Example Detection](
http://arxiv.org/abs/1803.08533)' paper. I've tried to clean it up a little to
make it clearer and easier to understand, and to remove code not relevant to
this particular paper.  It has a couple of requirements; you will need to
install [cleverhans](https://github.com/tensorflow/cleverhans/). Other than
that the requirements are basically the standard scipy stack, plus keras and
tensorflow.

### Reproducing ROC curves

#### Prepare dataset

The ASSIRA cats and dogs dataset used in the paper can be
downloaded
[here](https://www.microsoft.com/en-us/download/details.aspx?id=54765).
Save the zip file in repository's root directory and unzip it:

```bash
unzip [DATASET FILE].zip
```


#### Train the classifier

To train the ResNet-based cats and dogs classifier on the dataset execute:

```bash
python cats_and_dogs.py
```


#### Evaluate classifier on synthetic dataset

To evalute the classfier on a synthetic dataset as described in the paper execute:

```bash
python ROC_curves_cats.py
```

The script calculates ROC curves and their AUCs based on entropy and MI for the different models and saves that data as
`*.h5` files.


#### Plot results

To actually plot the ROC curves and some adversarial examples execute:

```bash
python plot_roc_cats.py [FILENAME].h5 # e.g. my_roc_curves_fgm.h5
```