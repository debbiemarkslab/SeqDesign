## Installation

We recommend using SeqDesign with a GPU that supports CUDA, especially for training.
If a GPU is available, install the [TensorFlow GPU dependencies](https://www.tensorflow.org/install/gpu), 
then install the SeqDesign dependencies with:
```shell script
pip install -r requirements_gpu.txt
```

Using the [linux_setup.sh](linux_setup.sh) script, 
installation on a fresh Ubuntu 18.04 LTS machine took 5 minutes.

If no GPU is available, use:  
```shell script
pip install -r requirements.txt
```

Then install SeqDesign:
```shell script
python setup.py install
```

### Used software and versions tested:
- python - 3.7  
- tensorflow - 1.15  
- numpy - 1.15  
- scipy - 0.19  
- sklearn - 0.18  

Tested on Ubuntu 18.04 LTS
