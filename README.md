This repository provides code needed to reproduce the experiments in my
'[Understanding Measures of Uncertainty for Adversarial Example Detection](
http://arxiv.org/abs/1803.08533)' paper. I've tried to clean it up a little to
make it clearer and easier to understand, and to remove code not relevant to
this particular paper.  It has a couple of requirements; you will need to
install [cleverhans](https://github.com/tensorflow/cleverhans/). Other than
that the requirements are basically the standard scipy stack, plus keras and
tensorflow.  The ASSIRA cats and dogs dataset used in the paper can be
downloaded
[here](https://www.microsoft.com/en-us/download/details.aspx?id=54765).
