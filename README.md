########  Sample-based Distributional Policy Gradient (SDPG) 
########  Submitted to CDC 2020

The SDPG implementation is built upon available D4PG code at https://github.com/msinto93/D4PG

## Usage

To select the environment and other hyperparameters such as learning rate and discount factor, modify "params.py"

The network architechture and loss functions can be found in "utils/network.py"

The training sequence can be found in "learner.py"



To train the SDPG network, run

  $ python train.py

This will train the network on the specified environment and periodically save checkpoints to the `/ckpts` folder.

To test the saved checkpoints during training, run

  $ python test_every_new_ckpt.py


Once the network is trained, we can visualise its performance in the environment by running

  $ python play.py


## Requirements

- OpenAI Gym
- MuJoCo
- TensorFlow
- Python3
- [numpy](http://www.numpy.org/)
- [scipy](http://www.scipy.org/install.html) 
- [opencv-python](http://opencv.org/)
- [imageio](http://imageio.github.io/) (requires [pillow](https://python-pillow.org/))
- [inotify-tools](https://github.com/rvoicilas/inotify-tools/wiki) 
