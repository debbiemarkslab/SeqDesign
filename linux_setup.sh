#!/bin/bash
# Example installation script from scratch on Linux
# Miniconda and Tensorflow 1.12 are installed here, but a working Tensorflow 1 environment can substitute.
# Before running this script, first run `git clone https://github.com/debbiemarkslab/SeqDesign.git`
# and then `cd SeqDesign`

#sudo apt update
#sudo apt install -y --no-install-recommends nvidia-driver-430
## Reboot.

# set up conda and the SeqDesign environment
wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
sh Miniconda2-latest-Linux-x86_64.sh -b -p ~/miniconda2
~/miniconda2/bin/conda init
bash
conda create -n seqdesign -y python=2.7 "tensorflow-gpu=1.12" scipy scikit-learn
conda activate seqdesign
python -c "from tensorflow.python.client import device_lib; print device_lib.list_local_devices()"  # test GPU install

# download SeqDesign code
# git clone https://github.com/debbiemarkslab/SeqDesign.git
cd SeqDesign || exit
python setup.py install  # use setup.py develop if you want to modify the code files

# download demo/example data
cd examples || exit
./download_example_data.sh

# # to run training demo:
# ./demo_train.sh

# # to run calc_logprobs using trained weights:
# ./demo_calc_logprobs.sh
