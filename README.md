This repository provides code needed to reproduce the experiments in my
'Understanding Measures of Uncertainty for Adversarial Example Detection'
paper. I've tried to clean it up somewhat from the workspace I used to do the
work (which included a lot of redundant scripts for other experiments we
haven't yet written up) to make it clearer and easier to understand. It has a
couple of requirements; you will need to install
[cleverhans](https://github.com/tensorflow/cleverhans/). Other than that the
requirements are basically the standard scipy stack, plus keras and
tensorflow. The ASSIRA cats and dogs dataset used in the paper can be
downloaded
[here](https://www.microsoft.com/en-us/download/details.aspx?id=54765).