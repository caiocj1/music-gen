# Symbol-Level Melody Completion

An attempt at implementing a GAN similar to the one explained in https://www.researchgate.net/publication/340640304_A_Symbol-level_Melody_Completion_Based_on_a_Convolutional_Neural_Network_with_Generative_Adversarial_Learning .

All credit to the authors of the paper:
- Kosuke Nakamura
- Takashi Nose
- Yuya Chiba
- Akinori Ito

The data set used for training consists of MAESTRO midi files: https://magenta.tensorflow.org/datasets/maestro.

To create the environment, run ``conda env create -f environment.yml``.

To track training, ``tensorboard --logdir lightning_logs``.

To launch training, ``python run_training.py -v version_name``.
