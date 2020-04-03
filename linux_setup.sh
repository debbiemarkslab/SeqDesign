#!/bin/bash
# Example installation script of SeqDesign for Tensorflow-GPU from scratch on Linux
# Miniconda and Tensorflow 1.12 are installed here, but a working Tensorflow 1 environment can substitute.
# Before running this script, first run `git clone https://github.com/debbiemarkslab/SeqDesign.git`
# and then `cd SeqDesign`
# If NVIDIA drivers have not been installed before, this script must be run twice, rebooting the system in between.

if [ ! -f "/proc/driver/nvidia/version" ]; then
  echo "NVIDIA driver not found; installing."
  sudo apt update
  sudo apt install -y --no-install-recommends nvidia-driver-430
  echo "
NVIDIA drivers installed.
Please reboot your system, then run linux_setup.sh a second time."
  exit
fi

# set up conda and the SeqDesign environment
wget https://repo.anaconda.com/miniconda/Miniconda2-latest-Linux-x86_64.sh
sh Miniconda2-latest-Linux-x86_64.sh -b -p ~/miniconda2
rm Miniconda2-latest-Linux-x86_64.sh
$HOME/miniconda2/bin/conda init
$HOME/miniconda2/bin/conda create -n seqdesign -y python=2.7 "tensorflow-gpu=1.12" scipy scikit-learn
$HOME/miniconda2/envs/seqdesign/bin/python -c "from tensorflow.python.client import device_lib; print device_lib.list_local_devices()"  # test GPU install

# download SeqDesign code
# git clone https://github.com/debbiemarkslab/SeqDesign.git
# cd SeqDesign || exit
$HOME/miniconda2/envs/seqdesign/bin/python setup.py install  # use setup.py develop if you want to modify the code files

# download demo/example data
cd examples || exit
./download_example_data.sh

echo "
SeqDesign installed.
Run 'source ~/.bashrc; conda activate seqdesign' before using."

# # to run training demo:
# ./demo_train.sh

# # to run calc_logprobs using trained weights:
# ./demo_calc_logprobs.sh
